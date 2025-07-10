use crate::ir::{errors::ValidationError, graph::Graph, types::NodeID};

pub struct GraphValidator<'a> {
    graph: &'a Graph,
}

struct ValidationContext {
    current_node: Option<NodeID>,
}

pub type ValidationResult = Result<(), Vec<ValidationError>>;

fn ok() -> ValidationResult {
    Ok(())
}

fn single_error(error: ValidationError) -> ValidationResult {
    Err(vec![error])
}

fn combine_results(results: Vec<ValidationResult>) -> ValidationResult {
    let mut all_errors = Vec::new();

    for result in results {
        if let Err(mut errors) = result {
            all_errors.append(&mut errors);
        }
    }

    if all_errors.is_empty() {
        return ok();
    } else {
        return Err(all_errors);
    }
}

impl ValidationContext {
    fn new() -> Self {
        ValidationContext { current_node: None }
    }

    fn set_current_node(&mut self, node_id: NodeID) {
        self.current_node = Some(node_id)
    }

    fn clear_current_node(&mut self) {
        self.current_node = None
    }
}

impl<'a> GraphValidator<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        GraphValidator { graph }
    }

    pub fn validate_node_references(&self) -> ValidationResult {
        todo!("Implement node reference validation")
    }

    pub fn detect_cycles(&self) -> ValidationResult {
        todo!("Implement cycle detection")
    }

    pub fn validate_operation_constraints(&self) -> ValidationResult {
        todo!("Implement operation constraint validation")
    }

    pub fn validate(&self) -> ValidationResult {
        todo!("Implement comprehensive validation")
    }
}
