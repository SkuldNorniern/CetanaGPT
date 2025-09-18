use cetana::{
    nn::{activation::Swish, Layer},
    tensor::Tensor,
    MlResult,
};

/// GELU (Gaussian Error Linear Unit) activation function
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let coeff = 0.044715;
        
        // Calculate x^3
        let x_cubed = x.pow_scalar(3.0)?;
        
        // Calculate inner term: sqrt(2/π) * (x + 0.044715 * x³)
        let inner = x.add(&x_cubed.mul_scalar(coeff)?)?;
        let inner = inner.mul_scalar(sqrt_2_over_pi)?;
        
        // Calculate tanh using the formula: (e^x - e^(-x)) / (e^x + e^(-x))
        let exp_pos = inner.exp()?;
        let exp_neg = inner.mul_scalar(-1.0)?.exp()?;
        let tanh_term = exp_pos.sub(&exp_neg)?.div(&exp_pos.add(&exp_neg)?)?;
        
        // Calculate final result: 0.5 * x * (1 + tanh(...))
        let one_plus_tanh = tanh_term.add_scalar(1.0)?;
        let result = x.mul(&one_plus_tanh)?.mul_scalar(0.5)?;
        
        Ok(result)
    }
}

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        // ReLU: max(0, x) - using a simple implementation
        let zero = Tensor::zeros_like(x)?;
        Ok(x.max(&zero).clone())
    }
}

/// Swish activation function (x * sigmoid(x))
pub struct SwishActivation {
    swish: Swish,
}

impl SwishActivation {
    pub fn new() -> Self {
        Self {
            swish: Swish,
        }
    }

    pub fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        self.swish.forward(x)
    }
}

/// SwiGLU (Swish-Gated Linear Unit) activation function
/// SwiGLU(x) = Swish(x) * Linear(x)
/// This is commonly used in modern transformer architectures like PaLM and LLaMA
pub struct SwiGLU {
    swish: Swish,
}

impl SwiGLU {
    pub fn new() -> Self {
        Self {
            swish: Swish,
        }
    }

    /// Forward pass for SwiGLU
    /// Input should have shape [..., 2 * hidden_size] where the first half is the gate
    /// and the second half is the value
    pub fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        let shape = x.shape();
        let hidden_size = shape[shape.len() - 1] / 2;
        
        // Split the input into gate and value parts
        let gate = x.slice(&[&[0..shape[0]], &[0..hidden_size]])?;
        let value = x.slice(&[&[0..shape[0]], &[hidden_size..2 * hidden_size]])?;
        
        // Apply Swish to the gate
        let swish_gate = self.swish.forward(&gate)?;
        
        // Element-wise multiplication: Swish(gate) * value
        let result = swish_gate.mul(&value)?;
        
        Ok(result)
    }
}
