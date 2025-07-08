use xyntra::ir::{
    graph::Graph,
    ops::Node,
    types::{NodeID, OpKind, TensorShape},
};

/// Creates a test NodeID with a known value for consistent testing
pub fn create_test_node_id() -> NodeID {
    NodeID::new(42)
}

/// Creates a test NodeID with a specific value
pub fn create_test_node_id_with_value(id: u32) -> NodeID {
    NodeID::new(id)
}

/// Creates a TensorShape from dimensions for cleaner test code
pub fn create_test_tensor_shape(dims: Vec<usize>) -> TensorShape {
    TensorShape::new(dims)
}

/// Creates a scalar TensorShape (empty dimensions)
pub fn create_scalar_tensor_shape() -> TensorShape {
    TensorShape::new(vec![])
}

/// Builds a simple 3-node graph: input → matmul → output
pub fn build_simple_graph() -> Graph {
    let mut graph = Graph::new();

    // Add input node (no inputs, one output)
    let input_id = graph.add_node(OpKind::Custom("Input".to_string()), vec![], vec![]);

    // Add matmul node (one input, one output)
    let matmul_id = graph.add_node(OpKind::MatMul, vec![input_id], vec![]);

    // Add output node (one input, no outputs)
    let _output_id = graph.add_node(
        OpKind::Custom("Output".to_string()),
        vec![matmul_id],
        vec![],
    );

    graph
}

/// Builds a more complex graph for advanced testing
pub fn build_complex_graph() -> Graph {
    let mut graph = Graph::new();

    // Create a matmul → gelu → dropout chain
    let input1_id = graph.add_node(OpKind::Custom("Input1".to_string()), vec![], vec![]);
    let input2_id = graph.add_node(OpKind::Custom("Input2".to_string()), vec![], vec![]);

    let matmul_id = graph.add_node(OpKind::MatMul, vec![input1_id, input2_id], vec![]);
    let gelu_id = graph.add_node(OpKind::Gelu, vec![matmul_id], vec![]);
    let dropout_id = graph.add_node(OpKind::Dropout, vec![gelu_id], vec![]);

    let _output_id = graph.add_node(
        OpKind::Custom("Output".to_string()),
        vec![dropout_id],
        vec![],
    );

    graph
}

/// Custom assertion for TensorShape equality with better error messages
pub fn assert_tensor_shapes_equal(expected: &TensorShape, actual: &TensorShape) {
    assert_eq!(
        expected, actual,
        "TensorShapes not equal. Expected: {:?}, Actual: {:?}",
        expected, actual
    );
}

/// Custom assertion for NodeID equality with better error messages
pub fn assert_node_ids_equal(expected: NodeID, actual: NodeID) {
    assert_eq!(
        expected, actual,
        "NodeIDs not equal. Expected: {:?}, Actual: {:?}",
        expected, actual
    );
}

/// Helper to create a test OpKind for consistent testing
pub fn create_test_op_kind() -> OpKind {
    OpKind::MatMul
}

/// Helper to create various OpKind variants for testing
pub fn create_all_op_kinds() -> Vec<OpKind> {
    vec![
        OpKind::MatMul,
        OpKind::Add,
        OpKind::Gelu,
        OpKind::Dropout,
        OpKind::Softmax,
        OpKind::LayerNorm,
        OpKind::Custom("TestOp".to_string()),
    ]
}
