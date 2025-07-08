use crate::ir::types::{NodeID, OpKind};

pub struct Node {
    pub id: NodeID,
    pub op: OpKind,
    pub inputs: Vec<NodeID>,
    pub outputs: Vec<NodeID>,
}

impl Node {
    pub fn new(id: NodeID, op: OpKind, inputs: Vec<NodeID>, outputs: Vec<NodeID>) -> Self {
        Node {
            id,
            op,
            inputs,
            outputs,
        }
    }

    pub fn id(&self) -> NodeID {
        self.id
    }

    pub fn op(&self) -> &OpKind {
        &self.op
    }

    pub fn inputs(&self) -> &Vec<NodeID> {
        &self.inputs
    }

    pub fn outputs(&self) -> &Vec<NodeID> {
        &self.outputs
    }
}
