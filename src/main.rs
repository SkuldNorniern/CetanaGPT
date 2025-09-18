mod config;
mod attention;
mod act;
mod mlp;
mod block;
mod model;
mod dataloader;
mod train;
mod logger;

use crate::{
    config::GPTConfig,
    train::{train_model, TrainingConfig},
    dataloader::SimpleTokenizer,
};
use log::{info, LevelFilter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    logger::init(LevelFilter::Info).map_err(|e| format!("Failed to initialize logger: {}", e))?;
    
    info!("Starting CetanaGPT demo");

    // Create model configuration
    let model_config = GPTConfig::gpt2_small();
    info!("Using GPT-2 small configuration");

    // Create training configuration
    let training_config = TrainingConfig {
        learning_rate: 1e-3,
        max_iters: 1000,
        eval_interval: 200,
        eval_iters: 50,
        grad_clip: 1.0,
        batch_size: 2,
        block_size: 8,
    };

    // Sample training text
    let train_text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    let val_text = "Once upon a time in a land far away, there lived a wise old wizard. ".repeat(10);

    info!("Training text length: {} characters", train_text.len());
    info!("Validation text length: {} characters", val_text.len());

    // Train the model
    info!("Starting training...");
    let stats = train_model(&model_config, &training_config, &train_text, Some(&val_text))?;
    
    info!("Training completed!");
    info!("Final training loss: {:.4}", stats.last().unwrap().loss);

    // Generate some text
    let _tokenizer = SimpleTokenizer::new().map_err(|e| format!("Failed to create tokenizer: {}", e))?;
    let _model = crate::model::GPT::new(&model_config)?;
    
    // Note: In a real implementation, you would load the trained model weights here
    // For this demo, we'll just show the generation interface
    
    let prompt = "The quick brown fox";
    info!("Generating text with prompt: '{}'", prompt);
    
    // This would work with a properly trained model:
    // let generated = generate_text(&mut model, &tokenizer, prompt, 50, 1.0, Some(5))?;
    // info!("Generated text: '{}'", generated);
    
    info!("Demo completed successfully!");
    Ok(())
}
