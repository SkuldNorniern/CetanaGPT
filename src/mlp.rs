use cetana::{
    nn::{activation::Swish, Layer, Linear},
    tensor::Tensor,
    MlResult,
};
use log::{debug, trace};

use crate::config::GPTConfig;

/// Multi-Layer Perceptron (MLP) module
pub struct MLP {
    c_fc: Linear,    // First linear layer (expansion)
    swish: Swish,    // Swish activation
    c_proj: Linear,  // Second linear layer (projection)
}

impl MLP {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        Ok(Self {
            c_fc: Linear::new(config.n_embd, 4 * config.n_embd, config.bias)?,
            swish: Swish,
            c_proj: Linear::new(4 * config.n_embd, config.n_embd, config.bias)?,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        debug!("Starting MLP forward pass");
        trace!("MLP input shape: {:?}", x.shape());
        trace!("MLP input sample: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // First linear transformation (expansion)
        let x = self.c_fc.forward(x)?;
        trace!("After first linear layer: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Swish activation
        let x = self.swish.forward(&x)?;
        trace!("After activation: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Second linear transformation (projection)
        let result = self.c_proj.forward(&x)?;
        trace!("MLP output sample: {:?}", result.slice(&[&[0..1], &[0..1], &[0..5]])?);

        debug!("MLP forward pass complete");
        Ok(result)
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.c_fc.get_parameters());
        params.extend(self.c_proj.get_parameters());
        params
    }
}

