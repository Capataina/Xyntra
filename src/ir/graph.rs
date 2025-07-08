use std::collections::HashMap;

use crate::ir::{
    ops::Node,
    types::{NodeID, OpKind},
};

#[derive(Default)]
pub struct Graph {
    nodes: HashMap<NodeID, Node>,
    next_id: u32,
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    pub fn add_node(&mut self, op: OpKind, inputs: Vec<NodeID>, outputs: Vec<NodeID>) -> NodeID {
        let new_node_id: NodeID = NodeID::new(self.next_id);
        self.next_id += 1;
        let new_node: Node = Node {
            id: new_node_id,
            op,
            inputs,
            outputs,
        };

        self.nodes.insert(new_node_id, new_node);
        new_node_id
    }

    pub fn get_node(&self, node_id: NodeID) -> Option<&Node> {
        self.nodes.get(&node_id)
    }
}
