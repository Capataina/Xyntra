use std::path::PathBuf;

use crate::ir::errors::{ValidationError, XyntraError};

struct XyntraConfig {
    input_file: Option<PathBuf>,
    output_dir: PathBuf,
    backend: BackendType,
    optimisation_level: u8,
    tile_size: usize,
    block_size: usize,
    enable_debug: bool,
    export_ir: bool,
}

#[derive(Debug, Default)]
enum BackendType {
    #[default]
    Wgsl,
    CudaPtx,
}

impl Default for XyntraConfig {
    fn default() -> Self {
        XyntraConfig {
            input_file: None,
            output_dir: PathBuf::from("."),
            backend: BackendType::default(),
            optimisation_level: 2,
            tile_size: 16,
            block_size: 256,
            enable_debug: false,
            export_ir: false,
        }
    }
}

impl XyntraConfig {
    pub fn validate(&self) -> Result<(), XyntraError> {
        if (self.tile_size != 0) && ((self.tile_size & (self.tile_size - 1)) != 0) {
            return Err(XyntraError::Validation(
                ValidationError::InvalidGPUParameter {
                    parameter: "tile_size".to_string(),
                    value: self.tile_size,
                    valid_range: "Must be a power of 2.".to_string(),
                },
            ));
        }

        if self.tile_size < 4 || self.tile_size > 64 {
            return Err(XyntraError::Validation(
                ValidationError::InvalidGPUParameter {
                    parameter: "tile_size".to_string(),
                    value: self.tile_size,
                    valid_range: "must be between 4 and 64".to_string(),
                },
            ));
        }

        if (self.block_size != 0) && ((self.block_size & (self.block_size - 1)) != 0) {
            return Err(XyntraError::Validation(
                ValidationError::InvalidGPUParameter {
                    parameter: "block_size".to_string(),
                    value: self.block_size,
                    valid_range: "must be a power of 2".to_string(),
                },
            ));
        }

        if self.block_size < 64 || self.block_size > 1024 {
            return Err(XyntraError::Validation(
                ValidationError::InvalidGPUParameter {
                    parameter: "block_size".to_string(),
                    value: self.block_size,
                    valid_range: "must be between 64 and 1024".to_string(),
                },
            ));
        }

        // Check optimization_level is 0-3
        if self.optimisation_level > 3 {
            return Err(XyntraError::Validation(
                ValidationError::InvalidConfigValue {
                    field: "optimization_level".to_string(),
                    value: self.optimisation_level.to_string(),
                    reason: "must be between 0 and 3".to_string(),
                },
            ));
        }

        if let Some(ref path) = self.input_file {
            if !path.exists() {
                return Err(XyntraError::Validation(ValidationError::InvalidFilePath {
                    path: path.display().to_string(),
                    reason: "file does not exist".to_string(),
                }));
            }
        }

        if std::fs::metadata(&self.output_dir).is_err() {
            return Err(XyntraError::Validation(ValidationError::InvalidFilePath {
                path: self.output_dir.display().to_string(),
                reason: "directory does not exist or is not accessible".to_string(),
            }));
        }

        Ok(())
    }
}
