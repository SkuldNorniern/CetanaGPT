use cetana::{
    nn::{Layer, LayerNorm},
    tensor::Tensor,
    MlResult,
};
use log::{debug, trace};

use crate::{
    attention::CausalSelfAttention,
    config::GPTConfig,
    mlp::MLP,
};

/// Transformer block containing self-attention and MLP
pub struct Block {
    ln_1: LayerNorm,           // Layer norm before attention
    attn: CausalSelfAttention, // Self-attention module
    ln_2: LayerNorm,           // Layer norm before MLP
    mlp: MLP,                  // MLP module
}

impl Block {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        Ok(Self {
            ln_1: LayerNorm::new(
                vec![config.n_embd],
                None,
                None,
                None,
            )?,
            attn: CausalSelfAttention::new(config)?,
            ln_2: LayerNorm::new(
                vec![config.n_embd],
                None,
                None,
                None,
            )?,
            mlp: MLP::new(config)?,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        debug!("Starting transformer block forward pass");
        trace!("Block input shape: {:?}", x.shape());
        trace!("Block input sample: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Pre-norm attention
        let norm_x = self.ln_1.forward(x)?;
        trace!("After ln_1: {:?}", norm_x.slice(&[&[0..1], &[0..1], &[0..5]])?);
        
        let attn_out = self.attn.forward(&norm_x)?;
        trace!("After attention: {:?}", attn_out.slice(&[&[0..1], &[0..1], &[0..5]])?);
        
        // Residual connection
        let x = x.add(&attn_out)?;
        trace!("After attention residual: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Pre-norm MLP
        let norm_x = self.ln_2.forward(&x)?;
        trace!("After ln_2: {:?}", norm_x.slice(&[&[0..1], &[0..1], &[0..5]])?);
        
        let mlp_out = self.mlp.forward(&norm_x)?;
        trace!("After MLP: {:?}", mlp_out.slice(&[&[0..1], &[0..1], &[0..5]])?);
        
        // Residual connection
        let result = x.add(&mlp_out)?;
        trace!("After MLP residual: {:?}", result.slice(&[&[0..1], &[0..1], &[0..5]])?);

        debug!("Transformer block forward pass complete");
        Ok(result)
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.ln_1.get_parameters());
        params.extend(self.attn.get_parameters());
        params.extend(self.ln_2.get_parameters());
        params.extend(self.mlp.get_parameters());
        params
    }
}

