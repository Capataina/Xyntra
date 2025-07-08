mod common;

use common::{create_all_op_kinds, create_test_node_id_with_value, create_test_op_kind};
use xyntra::ir::{ops::Node, types::OpKind};

#[test]
fn test_node_creation() {
    let node_id = create_test_node_id_with_value(1);
    let op_kind = create_test_op_kind();
    let inputs = vec![create_test_node_id_with_value(0)];
    let outputs = vec![create_test_node_id_with_value(2)];

    let node = Node::new(node_id, op_kind, inputs.clone(), outputs.clone());

    // Verify all fields are set correctly
    assert_eq!(node.id, node_id);
    assert_eq!(node.inputs, inputs);
    assert_eq!(node.outputs, outputs);

    // Verify the op field (we need to match since OpKind doesn't implement PartialEq)
    match (&node.op, &OpKind::MatMul) {
        (OpKind::MatMul, OpKind::MatMul) => (), // OK
        _ => panic!("OpKind not set correctly"),
    }
}

#[test]
fn test_node_accessor_methods() {
    let node_id = create_test_node_id_with_value(10);
    let op_kind = OpKind::Gelu;
    let inputs = vec![
        create_test_node_id_with_value(8),
        create_test_node_id_with_value(9),
    ];
    let outputs = vec![create_test_node_id_with_value(11)];

    let node = Node::new(node_id, op_kind, inputs.clone(), outputs.clone());

    // Test id() method
    assert_eq!(node.id(), node_id);

    // Test op() method
    match node.op() {
        OpKind::Gelu => (), // OK
        _ => panic!("op() method returned wrong OpKind"),
    }

    // Test inputs() method
    assert_eq!(node.inputs(), &inputs);
    assert_eq!(node.inputs().len(), 2);
    assert_eq!(node.inputs()[0], create_test_node_id_with_value(8));
    assert_eq!(node.inputs()[1], create_test_node_id_with_value(9));

    // Test outputs() method
    assert_eq!(node.outputs(), &outputs);
    assert_eq!(node.outputs().len(), 1);
    assert_eq!(node.outputs()[0], create_test_node_id_with_value(11));
}

#[test]
fn test_node_with_no_inputs() {
    let node_id = create_test_node_id_with_value(0);
    let op_kind = OpKind::Custom("Input".to_string());
    let inputs = vec![]; // No inputs (like an input node)
    let outputs = vec![create_test_node_id_with_value(1)];

    let node = Node::new(node_id, op_kind, inputs, outputs);

    assert_eq!(node.inputs().len(), 0);
    assert!(node.inputs().is_empty());
    assert_eq!(node.outputs().len(), 1);
}

#[test]
fn test_node_with_no_outputs() {
    let node_id = create_test_node_id_with_value(5);
    let op_kind = OpKind::Custom("Output".to_string());
    let inputs = vec![create_test_node_id_with_value(4)];
    let outputs = vec![]; // No outputs (like an output node)

    let node = Node::new(node_id, op_kind, inputs, outputs);

    assert_eq!(node.inputs().len(), 1);
    assert_eq!(node.outputs().len(), 0);
    assert!(node.outputs().is_empty());
}

#[test]
fn test_node_with_multiple_inputs_and_outputs() {
    let node_id = create_test_node_id_with_value(10);
    let op_kind = OpKind::Add;
    let inputs = vec![
        create_test_node_id_with_value(7),
        create_test_node_id_with_value(8),
        create_test_node_id_with_value(9),
    ];
    let outputs = vec![
        create_test_node_id_with_value(11),
        create_test_node_id_with_value(12),
    ];

    let node = Node::new(node_id, op_kind, inputs.clone(), outputs.clone());

    assert_eq!(node.inputs().len(), 3);
    assert_eq!(node.outputs().len(), 2);

    // Verify all inputs are correct
    for (i, expected_input) in inputs.iter().enumerate() {
        assert_eq!(node.inputs()[i], *expected_input);
    }

    // Verify all outputs are correct
    for (i, expected_output) in outputs.iter().enumerate() {
        assert_eq!(node.outputs()[i], *expected_output);
    }
}

#[test]
fn test_node_with_all_op_kinds() {
    let base_node_id = 100u32;

    for (i, op_kind) in create_all_op_kinds().into_iter().enumerate() {
        let node_id = create_test_node_id_with_value(base_node_id + i as u32);
        let inputs = vec![create_test_node_id_with_value(0)];
        let outputs = vec![create_test_node_id_with_value(999)];

        let node = Node::new(node_id, op_kind, inputs, outputs);

        // Verify node was created successfully
        assert_eq!(node.id(), node_id);
        assert_eq!(node.inputs().len(), 1);
        assert_eq!(node.outputs().len(), 1);

        // Each op kind should be stored correctly (we can't easily test equality
        // since OpKind doesn't implement PartialEq, but creation should succeed)
    }
}

#[test]
fn test_node_custom_op_kind() {
    let node_id = create_test_node_id_with_value(42);
    let custom_op_name = "MyCustomOperation";
    let op_kind = OpKind::Custom(custom_op_name.to_string());
    let inputs = vec![];
    let outputs = vec![];

    let node = Node::new(node_id, op_kind, inputs, outputs);

    // Test that custom op is stored correctly
    match node.op() {
        OpKind::Custom(name) => assert_eq!(name, custom_op_name),
        _ => panic!("Custom OpKind not stored correctly"),
    }
}

#[test]
fn test_node_edge_cases() {
    // Test node with same input and output IDs (could happen in some graph patterns)
    let node_id = create_test_node_id_with_value(5);
    let op_kind = OpKind::LayerNorm;
    let shared_id = create_test_node_id_with_value(4);
    let inputs = vec![shared_id];
    let outputs = vec![shared_id]; // Same ID used for input and output

    let node = Node::new(node_id, op_kind, inputs, outputs);

    assert_eq!(node.inputs()[0], node.outputs()[0]);
    assert_eq!(node.inputs().len(), 1);
    assert_eq!(node.outputs().len(), 1);
}

#[test]
fn test_node_immutability() {
    let node_id = create_test_node_id_with_value(1);
    let op_kind = OpKind::Softmax;
    let inputs = vec![create_test_node_id_with_value(0)];
    let outputs = vec![create_test_node_id_with_value(2)];

    let node = Node::new(node_id, op_kind, inputs, outputs);

    // Test that returned references don't allow mutation
    let inputs_ref = node.inputs();
    let outputs_ref = node.outputs();

    // These should be immutable references
    assert_eq!(inputs_ref.len(), 1);
    assert_eq!(outputs_ref.len(), 1);

    // The references should remain valid and consistent
    assert_eq!(node.inputs(), inputs_ref);
    assert_eq!(node.outputs(), outputs_ref);
}
