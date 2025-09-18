use cetana::{
    backend::Device,
    nn::{activation::Softmax, Dropout, Layer, Linear},
    tensor::Tensor,
    MlResult,
};
use log::{debug, info, trace};

use crate::config::GPTConfig;

/// Causal self-attention module
pub struct CausalSelfAttention {
    c_attn: Linear,      // Query, Key, Value projection
    c_proj: Linear,      // Output projection
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    n_head: usize,
    n_embd: usize,
    head_size: usize,
    bias: Option<Tensor>, // Causal mask
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig) -> MlResult<Self> {
        let head_size = config.n_embd / config.n_head;
        
        // Create causal mask (lower triangular matrix)
        let bias = Self::create_causal_mask(config.block_size)?;
        
        Ok(Self {
            c_attn: Linear::new(config.n_embd, 3 * config.n_embd, config.bias)?,
            c_proj: Linear::new(config.n_embd, config.n_embd, config.bias)?,
            attn_dropout: Dropout::new(config.dropout as f64),
            resid_dropout: Dropout::new(config.dropout as f64),
            n_head: config.n_head,
            n_embd: config.n_embd,
            head_size,
            bias: Some(bias),
        })
    }

    /// Create causal mask for preventing attention to future tokens
    fn create_causal_mask(block_size: usize) -> MlResult<Tensor> {
        let mut mask_data = vec![0.0; block_size * block_size];
        
        // Create lower triangular matrix (1s below diagonal, 0s on and above)
        for i in 0..block_size {
            for j in 0..block_size {
                if j > i {
                    mask_data[i * block_size + j] = f32::NEG_INFINITY;
                }
            }
        }
        
        Tensor::from_vec(
            mask_data,
            &[1, 1, block_size, block_size],
            std::sync::Arc::new(cetana::backend::CpuBackend::new()?),
        )
    }

    pub fn forward(&mut self, x: &Tensor) -> MlResult<Tensor> {
        info!("Starting causal self-attention computation");
        let shape = x.shape();
        let (b, t, c) = (shape[0], shape[1], shape[2]);
        
        debug!("Input shape - batch_size: {}, seq_len: {}, n_embd: {}", b, t, c);
        trace!("Input tensor sample: {:?}", x.slice(&[&[0..1], &[0..1], &[0..5]])?);

        // Calculate query, key, values for all heads in one go
        let qkv = self.c_attn.forward(x)?;
        trace!("QKV shape after projection: {:?}", qkv.shape());

        // Split into query, key, value
        let qkv_chunks = qkv.chunk(3, 2)?;
        let mut q = qkv_chunks[0].reshape(&[
            b as isize, 
            t as isize, 
            self.n_head as isize, 
            self.head_size as isize,
        ])?;
        let mut k = qkv_chunks[1].reshape(&[
            b as isize, 
            t as isize, 
            self.n_head as isize, 
            self.head_size as isize,
        ])?;
        let mut v = qkv_chunks[2].reshape(&[
            b as isize, 
            t as isize, 
            self.n_head as isize, 
            self.head_size as isize,
        ])?;

        trace!("After reshape - Q: {:?}, K: {:?}, V: {:?}", q.shape(), k.shape(), v.shape());

        // Transpose to get [B, nh, T, hs]
        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;
        v = v.transpose(1, 2)?;

        trace!("After transpose - Q: {:?}, K: {:?}, V: {:?}", q.shape(), k.shape(), v.shape());
        trace!("Q sample after transpose: {:?}", q.slice(&[&[0..1], &[0..1], &[0..1], &[0..5]])?);

        // Compute attention scores
        let k_t = k.transpose(-2, -1)?;
        trace!("K transpose shape: {:?}", k_t.shape());
        
        let att = q.matmul(&k_t)?;
        trace!("Raw attention scores shape: {:?}", att.shape());
        trace!("Attention scores sample: {:?}", att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?);

        // Scale by sqrt(head_size)
        let att = att.mul_scalar(1.0 / (self.head_size as f32).sqrt())?;
        trace!("Scaled attention scores sample: {:?}", att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?);

        // Apply causal mask
        if let Some(bias) = &self.bias {
            trace!("Applying causal mask");
            let mask = bias.slice(&[&[0..1], &[0..1], &[0..t], &[0..t]])?;
            trace!("Mask shape: {:?}", mask.shape());
            
            // Create a proper mask tensor for masked_fill
            let mask_tensor = mask.eq_scalar(0.0)?;
            let att = att.masked_fill(&mask_tensor, f32::NEG_INFINITY)?;
            trace!("Attention scores after masking sample: {:?}", att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?);
        }

        // Apply softmax
        let att = Softmax::new(Some(-1)).forward(&att)?;
        trace!("Softmax output sample: {:?}", att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?);

        // Apply dropout
        let att = self.attn_dropout.forward(&att)?;
        trace!("Attention scores after dropout sample: {:?}", att.slice(&[&[0..1], &[0..1], &[0..5], &[0..5]])?);

        // Apply attention to values
        let y = att.matmul(&v)?;
        trace!("After attention application: {:?}", y.shape());

        // Reshape back to [B, T, C]
        let y = y.transpose(1, 2)?.reshape(&[b as isize, t as isize, c as isize])?;

        // Output projection
        let y = self.c_proj.forward(&y)?;
        let y = self.resid_dropout.forward(&y)?;

        info!("Causal self-attention computation complete");
        Ok(y)
    }

    pub fn get_parameters(&self) -> Vec<(Tensor, Option<Tensor>)> {
        let mut params = Vec::new();
        params.extend(self.c_attn.get_parameters());
        params.extend(self.c_proj.get_parameters());
        params
    }
}

