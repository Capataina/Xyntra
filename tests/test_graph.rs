mod common;

use common::{build_complex_graph, build_simple_graph, create_test_node_id_with_value};
use xyntra::ir::{
    graph::Graph,
    types::{NodeID, OpKind},
};

#[test]
fn test_empty_graph_creation() {
    let graph = Graph::new();

    // Try to get a node that shouldn't exist
    let nonexistent_id = create_test_node_id_with_value(0);
    assert!(graph.get_node(nonexistent_id).is_none());

    // The first node added should get ID 0
    let mut graph = Graph::new();
    let first_id = graph.add_node(OpKind::MatMul, vec![], vec![]);
    assert_eq!(first_id.id(), 0);
}

#[test]
fn test_add_single_node() {
    let mut graph = Graph::new();

    let op_kind = OpKind::Gelu;
    let inputs = vec![];
    let outputs = vec![];

    let node_id = graph.add_node(op_kind, inputs.clone(), outputs.clone());

    // Verify the node was added and can be retrieved
    let retrieved_node = graph.get_node(node_id);
    assert!(retrieved_node.is_some());

    let node = retrieved_node.unwrap();
    assert_eq!(node.id(), node_id);
    assert_eq!(node.inputs(), &inputs);
    assert_eq!(node.outputs(), &outputs);

    // Verify the op kind (match since OpKind doesn't implement PartialEq)
    match node.op() {
        OpKind::Gelu => (), // OK
        _ => panic!("OpKind not stored correctly"),
    }
}

#[test]
fn test_add_multiple_nodes() {
    let mut graph = Graph::new();

    // Add first node
    let id1 = graph.add_node(OpKind::MatMul, vec![], vec![]);

    // Add second node
    let id2 = graph.add_node(OpKind::Add, vec![id1], vec![]);

    // Add third node
    let id3 = graph.add_node(OpKind::Gelu, vec![id2], vec![]);

    // Verify all nodes have unique, incrementing IDs
    assert_eq!(id1.id(), 0);
    assert_eq!(id2.id(), 1);
    assert_eq!(id3.id(), 2);

    // Verify all nodes can be retrieved
    assert!(graph.get_node(id1).is_some());
    assert!(graph.get_node(id2).is_some());
    assert!(graph.get_node(id3).is_some());

    // Verify node connections are correct
    let node2 = graph.get_node(id2).unwrap();
    assert_eq!(node2.inputs(), &vec![id1]);

    let node3 = graph.get_node(id3).unwrap();
    assert_eq!(node3.inputs(), &vec![id2]);
}

#[test]
fn test_get_nonexistent_node() {
    let graph = Graph::new();

    // Try to get nodes with various IDs that shouldn't exist
    let nonexistent_ids = vec![
        create_test_node_id_with_value(0),
        create_test_node_id_with_value(1),
        create_test_node_id_with_value(42),
        create_test_node_id_with_value(u32::MAX),
    ];

    for id in nonexistent_ids {
        assert!(graph.get_node(id).is_none());
    }
}

#[test]
fn test_graph_with_complex_connections() {
    let mut graph = Graph::new();

    // Create a diamond-shaped graph:
    //     input
    //    /     \
    //  op1     op2
    //    \     /
    //    output

    let input_id = graph.add_node(OpKind::Custom("Input".to_string()), vec![], vec![]);
    let op1_id = graph.add_node(OpKind::MatMul, vec![input_id], vec![]);
    let op2_id = graph.add_node(OpKind::Add, vec![input_id], vec![]);
    let output_id = graph.add_node(
        OpKind::Custom("Output".to_string()),
        vec![op1_id, op2_id],
        vec![],
    );

    // Verify the structure
    let input_node = graph.get_node(input_id).unwrap();
    assert!(input_node.inputs().is_empty());

    let op1_node = graph.get_node(op1_id).unwrap();
    assert_eq!(op1_node.inputs(), &vec![input_id]);

    let op2_node = graph.get_node(op2_id).unwrap();
    assert_eq!(op2_node.inputs(), &vec![input_id]);

    let output_node = graph.get_node(output_id).unwrap();
    assert_eq!(output_node.inputs(), &vec![op1_id, op2_id]);
}

#[test]
fn test_node_id_uniqueness() {
    let mut graph = Graph::new();
    let mut node_ids = Vec::new();

    // Add many nodes and collect their IDs
    for i in 0..100 {
        let op_kind = if i % 2 == 0 {
            OpKind::MatMul
        } else {
            OpKind::Add
        };
        let node_id = graph.add_node(op_kind, vec![], vec![]);
        node_ids.push(node_id);
    }

    // Verify all IDs are unique
    for i in 0..node_ids.len() {
        for j in (i + 1)..node_ids.len() {
            assert_ne!(
                node_ids[i], node_ids[j],
                "Found duplicate NodeIDs at indices {i} and {j}",
            );
        }
    }

    // Verify IDs are sequential
    for (i, node_id) in node_ids.iter().enumerate() {
        assert_eq!(node_id.id(), i as u32);
    }
}

#[test]
fn test_helper_function_simple_graph() {
    let graph = build_simple_graph();

    // The simple graph should have 3 nodes (input, matmul, output)
    let input_id = create_test_node_id_with_value(0);
    let matmul_id = create_test_node_id_with_value(1);
    let output_id = create_test_node_id_with_value(2);

    // Verify all nodes exist
    assert!(graph.get_node(input_id).is_some());
    assert!(graph.get_node(matmul_id).is_some());
    assert!(graph.get_node(output_id).is_some());

    // Verify connections
    let matmul_node = graph.get_node(matmul_id).unwrap();
    assert_eq!(matmul_node.inputs(), &vec![input_id]);

    let output_node = graph.get_node(output_id).unwrap();
    assert_eq!(output_node.inputs(), &vec![matmul_id]);
}

#[test]
fn test_helper_function_complex_graph() {
    let graph = build_complex_graph();

    // The complex graph should have 6 nodes total
    let input1_id = create_test_node_id_with_value(0);
    let input2_id = create_test_node_id_with_value(1);
    let matmul_id = create_test_node_id_with_value(2);
    let gelu_id = create_test_node_id_with_value(3);
    let dropout_id = create_test_node_id_with_value(4);
    let output_id = create_test_node_id_with_value(5);

    // Verify all nodes exist
    assert!(graph.get_node(input1_id).is_some());
    assert!(graph.get_node(input2_id).is_some());
    assert!(graph.get_node(matmul_id).is_some());
    assert!(graph.get_node(gelu_id).is_some());
    assert!(graph.get_node(dropout_id).is_some());
    assert!(graph.get_node(output_id).is_some());

    // Verify the op chain: matmul → gelu → dropout
    let matmul_node = graph.get_node(matmul_id).unwrap();
    assert_eq!(matmul_node.inputs(), &vec![input1_id, input2_id]);

    let gelu_node = graph.get_node(gelu_id).unwrap();
    assert_eq!(gelu_node.inputs(), &vec![matmul_id]);

    let dropout_node = graph.get_node(dropout_id).unwrap();
    assert_eq!(dropout_node.inputs(), &vec![gelu_id]);

    let output_node = graph.get_node(output_id).unwrap();
    assert_eq!(output_node.inputs(), &vec![dropout_id]);
}

#[test]
fn test_graph_node_retrieval_consistency() {
    let mut graph = Graph::new();

    let node_id = graph.add_node(OpKind::LayerNorm, vec![], vec![]);

    // Multiple calls to get_node should return the same reference
    let node1 = graph.get_node(node_id);
    let node2 = graph.get_node(node_id);

    assert!(node1.is_some());
    assert!(node2.is_some());

    // The references should point to the same data
    let node1_ptr = node1.unwrap() as *const _;
    let node2_ptr = node2.unwrap() as *const _;
    assert_eq!(node1_ptr, node2_ptr);
}

#[test]
fn test_graph_stress_test() {
    let mut graph = Graph::new();

    // Create a large linear chain of nodes
    let chain_length = 1000;
    let mut prev_id: Option<NodeID> = None;

    for i in 0..chain_length {
        let inputs = if let Some(prev) = prev_id {
            vec![prev]
        } else {
            vec![]
        };

        let op_kind = match i % 4 {
            0 => OpKind::MatMul,
            1 => OpKind::Gelu,
            2 => OpKind::Add,
            _ => OpKind::Dropout,
        };

        let node_id = graph.add_node(op_kind, inputs, vec![]);
        prev_id = Some(node_id);

        // Verify the node was added correctly
        assert!(graph.get_node(node_id).is_some());
        assert_eq!(node_id.id(), i as u32);
    }

    // Verify the entire chain is intact
    for i in 0..chain_length {
        let node_id = create_test_node_id_with_value(i as u32);
        let node = graph.get_node(node_id).unwrap();

        if i == 0 {
            // First node should have no inputs
            assert!(node.inputs().is_empty());
        } else {
            // Every other node should have exactly one input (the previous node)
            assert_eq!(node.inputs().len(), 1);
            assert_eq!(node.inputs()[0].id(), (i - 1) as u32);
        }
    }
}
