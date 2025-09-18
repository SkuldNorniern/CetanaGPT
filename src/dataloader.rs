use cetana::{
    backend::Device,
    tensor::Tensor,
    MlResult,
};
use log::{debug, info, trace};

/// Simple data loader for text data
pub struct DataLoader {
    data: Vec<u32>,
    batch_size: usize,
    block_size: usize,
    current_pos: usize,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(data: Vec<u32>, batch_size: usize, block_size: usize) -> Self {
        Self {
            data,
            batch_size,
            block_size,
            current_pos: 0,
        }
    }

    /// Get the next batch of data
    pub fn get_batch(&mut self) -> MlResult<Option<(Tensor, Tensor)>> {
        if self.current_pos + self.block_size >= self.data.len() {
            return Ok(None);
        }

        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();

        for _ in 0..self.batch_size {
            if self.current_pos + self.block_size >= self.data.len() {
                break;
            }

            let start = self.current_pos;
            let end = start + self.block_size;

            // Input sequence (tokens 0 to block_size-1)
            let x_seq = &self.data[start..end];
            x_batch.extend(x_seq);

            // Target sequence (tokens 1 to block_size)
            let y_seq = &self.data[start + 1..end + 1];
            y_batch.extend(y_seq);

            self.current_pos += 1;
        }

        if x_batch.is_empty() {
            return Ok(None);
        }

        let actual_batch_size = x_batch.len() / self.block_size;
        
        let x = Tensor::from_vec(
            x_batch.into_iter().map(|x: u32| x as f32).collect(),
            &[actual_batch_size, self.block_size],
            std::sync::Arc::new(cetana::backend::CpuBackend::new()?),
        )?;

        let y = Tensor::from_vec(
            y_batch.into_iter().map(|x: u32| x as f32).collect(),
            &[actual_batch_size, self.block_size],
            std::sync::Arc::new(cetana::backend::CpuBackend::new()?),
        )?;

        debug!("Generated batch - shape: {:?}", x.shape());
        trace!("X sample: {:?}", x.slice(&[&[0..1], &[0..5]])?);
        trace!("Y sample: {:?}", y.slice(&[&[0..1], &[0..5]])?);

        Ok(Some((x, y)))
    }

    /// Reset the data loader to the beginning
    pub fn reset(&mut self) {
        self.current_pos = 0;
        info!("Data loader reset");
    }

    /// Get the total number of batches available
    pub fn num_batches(&self) -> usize {
        if self.data.len() <= self.block_size {
            0
        } else {
            (self.data.len() - self.block_size).saturating_sub(self.current_pos)
        }
    }
}

/// Simple tokenizer for basic text processing
pub struct SimpleTokenizer {
    vocab: std::collections::HashMap<char, u32>,
    reverse_vocab: std::collections::HashMap<u32, char>,
    vocab_size: usize,
}

impl SimpleTokenizer {
    /// Create a new tokenizer
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut vocab = std::collections::HashMap::new();
        let mut reverse_vocab = std::collections::HashMap::new();
        
        // Add basic characters
        let mut idx = 0;
        
        // Add special tokens
        vocab.insert('\0', idx); // Padding token
        reverse_vocab.insert(idx, '\0');
        idx += 1;
        
        // Add printable ASCII characters
        for c in (32..127).map(|i| i as u8 as char) {
            vocab.insert(c, idx);
            reverse_vocab.insert(idx, c);
            idx += 1;
        }
        
        Ok(Self {
            vocab,
            reverse_vocab,
            vocab_size: idx as usize,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| self.vocab.get(&c).copied().unwrap_or(0))
            .collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        Ok(tokens
            .iter()
            .map(|&token| self.reverse_vocab.get(&token).copied().unwrap_or('\0'))
            .collect())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Create a simple dataset from text
pub fn create_dataset(text: &str, tokenizer: &SimpleTokenizer) -> Vec<u32> {
    info!("Creating dataset from text (length: {})", text.len());
    let tokens = tokenizer.encode(text);
    debug!("Dataset created with {} tokens", tokens.len());
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let tokenizer = SimpleTokenizer::new().unwrap();
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_dataloader() {
        let tokenizer = SimpleTokenizer::new().unwrap();
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let data = create_dataset(&text, &tokenizer);
        
        let mut loader = DataLoader::new(data, 2, 8);
        
        let batch = loader.get_batch().unwrap();
        assert!(batch.is_some());
        
        let (x, y) = batch.unwrap();
        assert_eq!(x.shape()[0], 2); // batch_size
        assert_eq!(x.shape()[1], 8); // block_size
        assert_eq!(y.shape()[0], 2); // batch_size
        assert_eq!(y.shape()[1], 8); // block_size
    }
}
