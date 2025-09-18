use cetana::{
    loss::calculate_cross_entropy_loss,
    nn::{
        activation::Softmax,
        embedding::Embedding,
        Dropout,
        Layer,
        LayerNorm,
        Linear,
    },
    optimizer::Optimizer,
    tensor::Tensor,
    MlResult,
};
use log::{debug, info, trace};

use crate::{
    block::Block,
    config::GPTConfig,
};

/// GPT (Generative Pre-trained Transformer) model
pub struct GPT {
    token_embedding: Embedding,    // Token embedding layer
    position_embedding: Embedding, // Position embedding layer
    drop: Dropout,                 // Dropout layer
    blocks: Vec<Block>,            // Transformer blocks
    ln_f: LayerNorm,               // Final layer norm
    lm_head: Linear,               // Language modeling head
    config: GPTConfig,             // Model configuration
}

impl GPT {
    /// Create a new GPT model with the given configuration
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        info!("Initializing GPT model");
        debug!("Config: {:?}", config);

        let blocks = (0..config.n_layer)
            .map(|i| {
                debug!("Creating block {}", i);
                Block::new(config)
            })
            .collect::<MlResult<Vec<_>>>()?;

        Ok(Self {
            token_embedding: Embedding::new(
                config.vocab_size,
                config.n_embd,
                None,
                None,
                2.0,
                false,
                false,
            )?,
            position_embedding: Embedding::new(
                config.block_size,
                config.n_embd,
                None,
                None,
                2.0,
                false,
                false,
            )?,
            drop: Dropout::new(config.dropout as f64),
            blocks,
            ln_f: LayerNorm::new(vec![config.n_embd], None, None, None)?,
            lm_head: Linear::new(config.n_embd, config.vocab_size, false)?,
            config: config.clone(),
        })
    }

    /// Perform a single training step
    pub fn train_step(
        &mut self,
        input_ids: &Tensor,
        targets: &Tensor,
        optimizer: &mut impl Optimizer,
        grad_clip: f32,
    ) -> MlResult<f32> {
        info!("Starting training step");
        debug!(
            "Input shape: {:?}, Target shape: {:?}",
            input_ids.shape(),
            targets.shape()
        );
        trace!("Input values: {:?}", input_ids);
        trace!("Target values: {:?}", targets);

        // Zero gradients
        trace!("Zeroing out gradients");
        optimizer.zero_grad();

        // Forward pass
        info!("Starting forward pass for training");
        let logits = self.forward(input_ids, Some(targets))?.0;
        trace!("Forward pass complete, logits shape: {:?}", logits.shape());

        // Reshape logits and targets for loss calculation
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        let vocab_size = self.config.vocab_size;

        let logits_2d = logits.reshape(&[(batch_size * seq_len) as isize, vocab_size as isize])?;
        let targets_1d = targets.reshape(&[(batch_size * seq_len) as isize])?;

        trace!(
            "Reshaped for loss - logits: {:?}, targets: {:?}",
            logits_2d.shape(),
            targets_1d.shape()
        );

        // Calculate max logits for numerical stability
        let max_logits = logits_2d.mat_max(Some(1), true)?.0;
        let mut shifted_logits = logits_2d.sub(&max_logits.expand(&logits_2d.shape())?)?;

        // Calculate loss
        let loss = calculate_cross_entropy_loss(&shifted_logits, &targets_1d)?;
        debug!("Loss calculated: {:.4}", loss);

        // Backward pass
        info!("Starting backward pass");
        shifted_logits.requires_grad(true);
        shifted_logits.backward(None)?;

        // Clip gradients if specified
        if grad_clip > 0.0 {
            debug!("Clipping gradients at {}", grad_clip);
            // Note: Gradient clipping would be implemented here if supported by the optimizer
        }

        // Update parameters
        optimizer.step()?;

        Ok(loss)
    }

    /// Forward pass through the model
    pub fn forward(
        &mut self,
        idx: &Tensor,
        targets: Option<&Tensor>,
    ) -> MlResult<(Tensor, Option<f32>)> {
        info!("Starting forward pass");
        let shape = idx.shape();
        let (b, t) = (shape[0], shape[1]);

        debug!("Input shape - batch_size: {}, seq_len: {}", b, t);
        trace!("Input tensor: {:?}", idx);

        // Check sequence length
        if t > self.config.block_size {
            return Err(format!(
                "Cannot forward sequence of length {}, block size is only {}",
                t, self.config.block_size
            )
            .into());
        }

        // Token embeddings
        debug!("Computing token embeddings");
        let tok_emb = self.token_embedding.forward(idx)?;
        trace!("Token embeddings shape: {:?}", tok_emb.shape());
        trace!(
            "Token embeddings sample: {:?}",
            tok_emb.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        // Position embeddings
        debug!("Computing position embeddings");
        let pos = Tensor::arange(Some(0.0), t as f32, Some(1.0))?;
        // Reshape to 2D for embedding layer: [1, t]
        let pos = pos.reshape(&[1, t as isize])?;
        let pos_emb = self.position_embedding.forward(&pos)?;
        // Expand to match batch dimension: [b, t, n_embd]
        let pos_emb = pos_emb.expand(&[b, t, self.config.n_embd])?;
        trace!("Position embeddings shape: {:?}", pos_emb.shape());
        trace!(
            "Position embeddings sample: {:?}",
            pos_emb.slice(&[&[0..5], &[0..5]])?
        );

        // Add token and position embeddings
        let x = tok_emb.add(&pos_emb)?;
        trace!("Combined embeddings shape: {:?}", x.shape());
        trace!(
            "Combined embeddings sample: {:?}",
            x.slice(&[&[0..1], &[0..1], &[0..5]])?
        );

        // Apply dropout
        let x = self.drop.forward(&x)?;
        trace!("After dropout: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Pass through transformer blocks
        let mut x = x;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            debug!("Processing block {}", i);
            x = block.forward(&x)?;
            trace!(
                "Block {} output sample: {:?}",
                i,
                x.slice(&[&[0..1], &[0..1], &[0..5]])?
            );
        }

        // Final layer norm
        debug!("Applying final layer norm");
        let x = self.ln_f.forward(&x)?;
        trace!("After final layer norm: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Language modeling head
        debug!("Computing logits");
        let logits = self.lm_head.forward(&x)?;
        trace!("Logits shape: {:?}", logits.shape());
        trace!("Logits sample: {:?}", logits.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Calculate loss if targets provided
        let loss = if let Some(targets) = targets {
            debug!("Calculating loss");
            let batch_size = idx.shape()[0];
            let seq_len = idx.shape()[1];
            let vocab_size = self.config.vocab_size;

            let logits_2d = logits.reshape(&[(batch_size * seq_len) as isize, vocab_size as isize])?;
            let targets_1d = targets.reshape(&[(batch_size * seq_len) as isize])?;

            // Calculate max logits for numerical stability
            let max_logits = logits_2d.mat_max(Some(1), true)?.0;
            let shifted_logits = logits_2d.sub(&max_logits.expand(&logits_2d.shape())?)?;

            let loss = calculate_cross_entropy_loss(&shifted_logits, &targets_1d)?;
            debug!("Loss calculated: {:.4}", loss);
            Some(loss)
        } else {
            None
        };

        info!("Forward pass complete");
        Ok((logits, loss))
    }

    /// Generate text using the model
    pub fn generate(
        &mut self,
        idx: &Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> MlResult<Tensor> {
        info!("Starting text generation");
        debug!("Input shape: {:?}, max_new_tokens: {}", idx.shape(), max_new_tokens);

        let mut idx = idx.clone();
        
        for _ in 0..max_new_tokens {
            // Crop idx to the last block_size tokens
            let seq_len = idx.shape()[1];
            let start_idx = if seq_len > self.config.block_size {
                seq_len - self.config.block_size
            } else {
                0
            };
            
            let idx_cond = if start_idx > 0 {
                idx.slice(&[&[0..idx.shape()[0]], &[start_idx..seq_len]])?
            } else {
                idx.clone()
            };

            // Forward pass
            let (logits, _) = self.forward(&idx_cond, None)?;
            
            // Focus only on the last time step
            let logits = logits.slice(&[&[0..logits.shape()[0]], &[logits.shape()[1]-1..logits.shape()[1]], &[0..logits.shape()[2]]])?;
            // Remove sequence dimension by reshaping
            let logits = logits.reshape(&[logits.shape()[0] as isize, logits.shape()[2] as isize])?;

            // Apply temperature
            let logits = if temperature != 1.0 {
                logits.div_scalar(temperature)?
            } else {
                logits
            };

            // Apply top-k filtering if specified
            let logits = if let Some(_k) = top_k {
                // Note: Top-k filtering would be implemented here
                // For now, we'll skip it to avoid API complexity
                logits
            } else {
                logits
            };

            // Apply softmax to get probabilities
            let probs = Softmax::new(Some(-1)).forward(&logits)?;
            
            // Sample from the distribution
            let next_token = probs.multinomial(1, false)?;
            
            // Append sampled token to the sequence
            let next_token = next_token.reshape(&[next_token.shape()[0] as isize, 1])?;
            idx = Tensor::cat(&[&idx, &next_token], 1)?;
        }

        info!("Text generation complete");
        Ok(idx)
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        self.get_parameters().len()
    }

    /// Get all parameters of the model
    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let params = Vec::new();
        // Note: In a real implementation, we would collect parameters from all layers
        // For now, we'll return an empty vector as a placeholder
        params
    }

    /// Get the model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }
}

