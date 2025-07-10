use core::fmt;

#[derive(Debug)]
pub enum XyntraError {
    // Errors that we can recover from (potentially)
    Validation(ValidationError),
    Parsing(ParsingError),

    // Fatal errors, stop execution
    System(SystemError),
    Internal(InternalError),
}

#[derive(Debug)]
pub enum ValidationError {
    InvalidTensorShape {
        expected: String,
        found: String,
    },
    IncompatibleShapes {
        op: String,
        shapes: Vec<String>,
    },
    InvalidNodeConnection {
        from: u32,
        to: u32,
        reason: String,
    },
    CyclicGraph {
        cycle_path: Vec<u32>,
    },
    MissingNode {
        node_id: u32,
    },
    InvalidOpInputCount {
        op: String,
        expected: usize,
        found: usize,
    },
    InvalidConfigValue {
        field: String,
        value: String,
        reason: String,
    },
    InvalidFilePath {
        path: String,
        reason: String,
    },
    InvalidGPUParameter {
        parameter: String,
        value: usize,
        valid_range: String,
    },
}

#[derive(Debug)]
pub enum ParsingError {
    InvalidFormat { format: String, reason: String },
    MalformedOnnx { details: String },
    UnsupportedOperation { op_name: String },
    CorruptedFile { file_path: String },
    MissingRequiredField { field: String },
}

#[derive(Debug)]
pub enum SystemError {
    OutOfMemory { requested: usize },
    GpuUnavailable { reason: String },
    FileNotFound { path: String },
    PermissionDenied { operation: String },
}

#[derive(Debug)]
pub enum InternalError {
    AssertionFailed { message: String },
    UnexpectedNone { context: String },
    InvalidState { expected: String, actual: String },
    NotImplemented { feature: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidTensorShape { expected, found } => {
                write!(
                    f,
                    "Invalid tensor shape: expected {expected} but found {found}.",
                )
            }

            ValidationError::IncompatibleShapes { op, shapes } => {
                write!(
                    f,
                    "Incompatible tensor shapes for operation '{}': shapes are {}.",
                    op,
                    shapes.join(", "),
                )
            }

            ValidationError::InvalidNodeConnection { from, to, reason } => {
                write!(
                    f,
                    "Invalid connection from node {from} to node {to}: {reason}.",
                )
            }

            ValidationError::CyclicGraph { cycle_path } => {
                write!(
                    f,
                    "Cyclic dependency detected in graph: nodes {}.",
                    cycle_path
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            }

            ValidationError::MissingNode { node_id } => {
                write!(f, "Referenced node {node_id} does not exist in the graph.",)
            }

            ValidationError::InvalidOpInputCount {
                op,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Operation '{op}' expects {expected} inputs but received {found}."
                )
            }

            ValidationError::InvalidConfigValue {
                field,
                value,
                reason,
            } => {
                write!(
                    f,
                    "Invalid configuration value for '{field}': '{value}' is not valid because {reason}."
                )
            }

            ValidationError::InvalidFilePath { path, reason } => {
                write!(f, "Invalid file path '{path}': {reason}.")
            }

            ValidationError::InvalidGPUParameter {
                parameter,
                value,
                valid_range,
            } => {
                write!(
                    f,
                    "Invalid GPU parameter '{parameter}': {value} is outside valid range {valid_range}."
                )
            }
        }
    }
}
