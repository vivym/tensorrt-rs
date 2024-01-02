use crate::{ffi, logger::Logger};
use cxx::UniquePtr;
use cuda_rs::stream::CuStream;

pub struct Runtime {
    pub(crate) runtime: UniquePtr<ffi::Runtime>,
    logger: Logger,
}

impl Runtime {
    pub fn new() -> Option<Self> {
        let mut logger = Logger::new();
        let runtime = ffi::create_runtime(logger.0.pin_mut());
        if runtime.is_null() {
            None
        } else {
            Some(Self { runtime, logger })
        }
    }

    pub fn logger(&mut self) -> &mut Logger {
        &mut self.logger
    }

    pub fn deserialize(&mut self, data: &[u8], size: usize) -> Option<CudaEngine> {
        let engine = self.runtime.pin_mut().deserialize(data, size);
        if engine.is_null() {
            None
        } else {
            Some(CudaEngine(engine))
        }
    }
}

pub struct CudaEngine(pub(crate) UniquePtr<ffi::CudaEngine>);

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TensorIOMode {
    // Tensor is not an input or output.
    NONE = 0,
    // Tensor is input to the engine.
    INPUT = 1,
    // Tensor is output by the engine.
    OUTPUT = 2
}

impl CudaEngine {
    pub fn create_execution_context(&mut self) -> Option<ExecutionContext> {
        let context = self.0.pin_mut().create_execution_context();
        if context.is_null() {
            None
        } else {
            Some(ExecutionContext(context))
        }
    }

    pub fn get_num_io_tensors(&self) -> i32 {
        self.0.get_num_io_tensors()
    }

    pub fn get_io_tensor_name(&self, index: i32) -> &str {
        self.0.get_io_tensor_name(index)
    }

    pub fn get_tensor_io_mode(&self, name: &str) -> TensorIOMode {
        match self.0.get_tensor_io_mode(name) {
            0 => TensorIOMode::NONE,
            1 => TensorIOMode::INPUT,
            2 => TensorIOMode::OUTPUT,
            mode => panic!("Invalid tensor io mode: {}", mode),
        }
    }

    pub fn get_tensor_shape(&self, name: &str) -> Vec<i32> {
        self.0.get_tensor_shape(name)
    }
}

pub struct ExecutionContext(pub(crate) UniquePtr<ffi::ExecutionContext>);

impl ExecutionContext {
    pub fn set_input_shape(&mut self, name: &str, shape: &[i32]) -> bool {
        self.0.pin_mut().set_input_shape(name, shape)
    }

    pub fn all_input_dimensions_specified(&self) -> bool {
        self.0.all_input_dimensions_specified()
    }

    pub fn set_tensor_address(&mut self, name: &str, address: usize) -> bool {
        self.0.pin_mut().set_tensor_address(name, address)
    }

    pub fn enqueue_v3(&mut self, stream: &CuStream) -> bool {
        let stream_raw = unsafe { stream.get_raw() };
        self.0.pin_mut().enqueue_v3(stream_raw as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logger::Severity;

    #[test]
    fn test_runtime() {
        use std::{io::Read, path::Path};

        let mut runtime = Runtime::new().unwrap();
        runtime.logger().log(Severity::Info, "Hello, world!");

        let engine_path = Path::new("../tmp/pp-ocr-v4-det-fp16.engine");
        if engine_path.exists() {
            let mut file = std::fs::File::open(engine_path).unwrap();
            let mut data = Vec::new();
            file.read_to_end(&mut data).unwrap();

            let mut engine = runtime.deserialize(data.as_slice(), data.len()).unwrap();
            let _context = engine.create_execution_context().unwrap();

            let num_io_tensors = engine.get_num_io_tensors();

            let msg = format!("num_io_tensors: {}", num_io_tensors);
            runtime.logger().log(Severity::Info, msg.as_str());

            for i in 0..num_io_tensors {
                let name = engine.get_io_tensor_name(i);
                let mode = engine.get_tensor_io_mode(name);
                let shape = engine.get_tensor_shape(name);
                let msg = format!("name: {}, mode: {:?}, shape: {:?}", name, mode, shape);
                runtime.logger().log(Severity::Info, msg.as_str());
            }
        } else {
            runtime.logger().log(Severity::Info, "Engine file not found! Skip test!");
        }
    }
}
