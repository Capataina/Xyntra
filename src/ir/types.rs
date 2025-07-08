#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeID(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape(Vec<usize>);

pub enum OpKind {
    MatMul,
    Add,
    Gelu,
    Dropout,
    Softmax,
    LayerNorm,
    Custom(String),
}

impl NodeID {
    pub fn new(id: u32) -> Self {
        NodeID(id)
    }

    pub fn id(&self) -> u32 {
        self.0
    }
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        TensorShape(dims)
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> usize {
        let mut final_size: usize = 1;
        for num in self.0.iter() {
            final_size *= num
        }
        final_size
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }
}
