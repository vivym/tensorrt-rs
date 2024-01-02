use thiserror::Error;

#[derive(Error, Debug)]
pub enum TRTError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Cuda error: {0}")]
    CudaError(#[from] cuda_rs::error::CuError),
    #[error("TensorRT runtime creation error")]
    RuntimeCreationError,
    #[error("TensorRT engine deserialization error")]
    EngineDeserializationError,
    #[error("TensorRT engine creation error")]
    EngineCreationError,
    #[error("TensorRT execution context not initialized")]
    ExecutionContextNotInitialized,
    #[error("TensorRT execution context creation error")]
    ExecutionContextCreationError,
    #[error("TensorRT invalid shape: {0:?}")]
    ShapeError(Vec<i32>),
    #[error("TensorRT invalid address")]
    InvalidAddress,
    #[error("TensorRT enqueue error")]
    EnqueueError,
    #[error("TensorRT reset shapes error")]
    ResetShapesError,
    #[error("TensorRT shape mismatch")]
    ShapeMismatch,
    #[error("TensorRT dtype mismatch")]
    DTypeMismatch,
}

pub type TRTResult<T> = Result<T, TRTError>;
