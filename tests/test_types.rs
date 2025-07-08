mod common;

use common::{create_scalar_tensor_shape, create_test_node_id, create_test_tensor_shape};
use std::collections::HashSet;
use xyntra::ir::types::NodeID;

#[test]
fn test_node_id_creation_and_equality() {
    let id1 = NodeID::new(42);
    let id2 = NodeID::new(42);
    let id3 = NodeID::new(43);

    // Test equality
    assert_eq!(id1, id2);
    assert_ne!(id1, id3);

    // Test ID retrieval
    assert_eq!(id1.id(), 42);
    assert_eq!(id3.id(), 43);

    // Test hash consistency (important for HashMap usage)
    let mut set = HashSet::new();
    set.insert(id1);
    assert!(set.contains(&id2)); // Should find id2 since it equals id1
    assert!(!set.contains(&id3)); // Should not find id3
}

#[test]
fn test_node_id_hash_consistency() {
    let id1 = NodeID::new(100);
    let id2 = NodeID::new(100);

    // If two NodeIDs are equal, they must have the same hash
    assert_eq!(id1, id2);

    let mut set = HashSet::new();
    set.insert(id1);
    set.insert(id2); // Should not increase set size since they're equal

    assert_eq!(set.len(), 1);
}

#[test]
fn test_tensor_shape_rank_calculation() {
    // Test scalar (0D)
    let scalar = create_scalar_tensor_shape();
    assert_eq!(scalar.rank(), 0);

    // Test 1D tensor
    let vector = create_test_tensor_shape(vec![10]);
    assert_eq!(vector.rank(), 1);

    // Test 2D tensor (matrix)
    let matrix = create_test_tensor_shape(vec![3, 4]);
    assert_eq!(matrix.rank(), 2);

    // Test 3D tensor
    let tensor_3d = create_test_tensor_shape(vec![2, 3, 4]);
    assert_eq!(tensor_3d.rank(), 3);

    // Test 4D tensor (common in deep learning: batch, channels, height, width)
    let tensor_4d = create_test_tensor_shape(vec![1, 3, 224, 224]);
    assert_eq!(tensor_4d.rank(), 4);
}

#[test]
fn test_tensor_shape_size_calculation() {
    // Test scalar (size should be 1)
    let scalar = create_scalar_tensor_shape();
    assert_eq!(scalar.size(), 1);

    // Test 1D tensor
    let vector = create_test_tensor_shape(vec![10]);
    assert_eq!(vector.size(), 10);

    // Test 2D tensor
    let matrix = create_test_tensor_shape(vec![3, 4]);
    assert_eq!(matrix.size(), 12); // 3 * 4 = 12

    // Test 3D tensor
    let tensor_3d = create_test_tensor_shape(vec![2, 3, 4]);
    assert_eq!(tensor_3d.size(), 24); // 2 * 3 * 4 = 24

    // Test with ones (should not affect size)
    let tensor_with_ones = create_test_tensor_shape(vec![1, 5, 1, 3, 1]);
    assert_eq!(tensor_with_ones.size(), 15); // 1 * 5 * 1 * 3 * 1 = 15

    // Test edge case: dimension with zero
    let tensor_with_zero = create_test_tensor_shape(vec![2, 0, 3]);
    assert_eq!(tensor_with_zero.size(), 0); // 2 * 0 * 3 = 0
}

#[test]
fn test_tensor_shape_scalar_detection() {
    // Test scalar detection
    let scalar = create_scalar_tensor_shape();
    assert!(scalar.is_scalar());

    // Test non-scalar detection
    let vector = create_test_tensor_shape(vec![1]);
    assert!(!vector.is_scalar()); // Even [1] is not a scalar

    let matrix = create_test_tensor_shape(vec![1, 1]);
    assert!(!matrix.is_scalar()); // [1, 1] is not a scalar

    let tensor = create_test_tensor_shape(vec![2, 3, 4]);
    assert!(!tensor.is_scalar());
}

#[test]
fn test_tensor_shape_equality_and_hashing() {
    let shape1 = create_test_tensor_shape(vec![2, 3, 4]);
    let shape2 = create_test_tensor_shape(vec![2, 3, 4]);
    let shape3 = create_test_tensor_shape(vec![2, 3, 5]);

    // Test equality
    assert_eq!(shape1, shape2);
    assert_ne!(shape1, shape3);

    // Test hash consistency
    let mut set = HashSet::new();
    set.insert(shape1.clone());
    assert!(set.contains(&shape2));
    assert!(!set.contains(&shape3));
}

#[test]
fn test_tensor_shape_edge_cases() {
    // Test very large tensor
    let large_tensor = create_test_tensor_shape(vec![1000, 1000]);
    assert_eq!(large_tensor.size(), 1_000_000);
    assert_eq!(large_tensor.rank(), 2);
    assert!(!large_tensor.is_scalar());

    // Test single dimension with large value
    let long_vector = create_test_tensor_shape(vec![1_000_000]);
    assert_eq!(long_vector.size(), 1_000_000);
    assert_eq!(long_vector.rank(), 1);
    assert!(!long_vector.is_scalar());
}

#[test]
fn test_helper_functions() {
    // Test our helper functions work correctly
    let test_id = create_test_node_id();
    assert_eq!(test_id.id(), 42);

    let test_shape = create_test_tensor_shape(vec![2, 3]);
    assert_eq!(test_shape.rank(), 2);
    assert_eq!(test_shape.size(), 6);

    let scalar = create_scalar_tensor_shape();
    assert!(scalar.is_scalar());
}
