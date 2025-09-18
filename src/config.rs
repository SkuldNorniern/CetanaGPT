#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub block_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
    pub bias: bool,
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 50304,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.1,
            bias: true,
        }
    }
}

impl GPTConfig {
    pub fn gpt2_small() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 50304,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.1,
            bias: true,
        }
    }

    pub fn gpt2_medium() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 50304,
            n_layer: 24,
            n_head: 16,
            n_embd: 1024,
            dropout: 0.1,
            bias: true,
        }
    }

    pub fn gpt2_large() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 50304,
            n_layer: 36,
            n_head: 20,
            n_embd: 1280,
            dropout: 0.1,
            bias: true,
        }
    }

    pub fn gpt2_xl() -> Self {
        Self {
            block_size: 1024,
            vocab_size: 50304,
            n_layer: 48,
            n_head: 25,
            n_embd: 1600,
            dropout: 0.1,
            bias: true,
        }
    }
}
