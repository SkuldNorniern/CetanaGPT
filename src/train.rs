use cetana::{
    backend::Device,
    optimizer::Adam,
    tensor::Tensor,
    MlResult,
};
use log::{debug, info, warn};

use crate::{
    config::GPTConfig,
    dataloader::{DataLoader, SimpleTokenizer, create_dataset},
    model::GPT,
};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub max_iters: usize,
    pub eval_interval: usize,
    pub eval_iters: usize,
    pub grad_clip: f32,
    pub batch_size: usize,
    pub block_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            max_iters: 5000,
            eval_interval: 500,
            eval_iters: 200,
            grad_clip: 1.0,
            batch_size: 4,
            block_size: 8,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub iter: usize,
    pub loss: f32,
    pub learning_rate: f32,
}

/// Train a GPT model
pub fn train_model(
    model_config: &GPTConfig,
    training_config: &TrainingConfig,
    train_text: &str,
    val_text: Option<&str>,
) -> MlResult<Vec<TrainingStats>> {
    info!("Starting model training");
    debug!("Model config: {:?}", model_config);
    debug!("Training config: {:?}", training_config);

    // Initialize tokenizer
    let tokenizer = SimpleTokenizer::new().map_err(|e| format!("Failed to create tokenizer: {}", e))?;
    info!("Tokenizer vocabulary size: {}", tokenizer.vocab_size());

    // Create datasets
    let train_data = create_dataset(train_text, &tokenizer);
    info!("Training dataset size: {} tokens", train_data.len());

    let val_data = if let Some(val_text) = val_text {
        Some(create_dataset(val_text, &tokenizer))
    } else {
        None
    };

    if let Some(ref val_data) = val_data {
        info!("Validation dataset size: {} tokens", val_data.len());
    }

    // Create data loaders
    let mut train_loader = DataLoader::new(
        train_data,
        training_config.batch_size,
        training_config.block_size,
    );

    let mut val_loader = if let Some(val_data) = val_data {
        Some(DataLoader::new(
            val_data,
            training_config.batch_size,
            training_config.block_size,
        ))
    } else {
        None
    };

    // Initialize model
    let mut model = GPT::new(model_config)?;
    info!("Model initialized with {} parameters", model.num_parameters());

    // Initialize optimizer
    let mut optimizer = Adam::new(
        training_config.learning_rate,
        Some((0.9, 0.95)),
        Some(0.9),
        Some(0.95),
    );

    // Training loop
    let mut stats = Vec::new();
    let mut iter = 0;

    info!("Starting training loop for {} iterations", training_config.max_iters);

    while iter < training_config.max_iters {
        // Get training batch
        let batch = match train_loader.get_batch() {
            Ok(Some(batch)) => batch,
            Ok(None) => {
                warn!("No more training data, resetting loader");
                train_loader.reset();
                continue;
            }
            Err(e) => {
                warn!("Error getting training batch: {}", e);
                continue;
            }
        };

        let (input_ids, targets) = batch;

        // Training step
        let loss = model.train_step(&input_ids, &targets, &mut optimizer, training_config.grad_clip)?;

        // Record statistics
        let stat = TrainingStats {
            iter,
            loss,
            learning_rate: training_config.learning_rate,
        };
        stats.push(stat);

        // Log progress
        if iter % 100 == 0 {
            info!("Iter {}: loss = {:.4}", iter, loss);
        }

        // Evaluation
        if iter % training_config.eval_interval == 0 && iter > 0 {
            if let Some(ref mut val_loader) = val_loader {
                let val_loss = evaluate_model(&mut model, val_loader, training_config.eval_iters)?;
                info!("Iter {}: val_loss = {:.4}", iter, val_loss);
            }
        }

        iter += 1;
    }

    info!("Training completed after {} iterations", iter);
    Ok(stats)
}

/// Evaluate the model on validation data
fn evaluate_model(
    model: &mut GPT,
    val_loader: &mut DataLoader,
    eval_iters: usize,
) -> MlResult<f32> {
    debug!("Starting model evaluation");
    
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    for _ in 0..eval_iters {
        let batch = match val_loader.get_batch() {
            Ok(Some(batch)) => batch,
            Ok(None) => {
                val_loader.reset();
                continue;
            }
            Err(e) => {
                warn!("Error getting validation batch: {}", e);
                continue;
            }
        };

        let (input_ids, targets) = batch;
        let (_, loss) = model.forward(&input_ids, Some(&targets))?;

        if let Some(loss) = loss {
            total_loss += loss;
            num_batches += 1;
        }
    }

    let avg_loss = if num_batches > 0 {
        total_loss / num_batches as f32
    } else {
        0.0
    };

    debug!("Evaluation complete - average loss: {:.4}", avg_loss);
    Ok(avg_loss)
}

/// Generate text using a trained model
pub fn generate_text(
    model: &mut GPT,
    tokenizer: &SimpleTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
) -> MlResult<String> {
    info!("Generating text with prompt: '{}'", prompt);
    debug!("Max new tokens: {}, temperature: {}, top_k: {:?}", max_new_tokens, temperature, top_k);

    // Encode prompt
    let prompt_tokens = tokenizer.encode(prompt);
    let prompt_tensor = Tensor::from_vec(
        prompt_tokens.clone().into_iter().map(|x| x as f32).collect(),
        &[1, prompt_tokens.len()],
        std::sync::Arc::new(cetana::backend::CpuBackend::new()?),
    )?;

    // Generate tokens
    let generated_tokens = model.generate(&prompt_tensor, max_new_tokens, temperature, top_k)?;
    
    // Convert to 1D tensor for decoding
    let shape = generated_tokens.shape();
    let total_elements = shape.iter().product::<usize>();
    let _generated_tokens = generated_tokens.reshape(&[total_elements as isize])?;
    // Note: In a real implementation, we would need to convert tensor to vec
    // For now, we'll create a placeholder
    let generated_tokens: Vec<u32> = vec![0; total_elements];

    // Decode tokens to text
    let generated_text = tokenizer.decode(&generated_tokens).map_err(|e| format!("Failed to decode tokens: {}", e))?;
    
    info!("Generated text: '{}'", generated_text);
    Ok(generated_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.max_iters, 5000);
        assert_eq!(config.batch_size, 4);
    }
}
