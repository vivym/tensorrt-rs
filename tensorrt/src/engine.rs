use crate::{
    error::{TRTError, TRTResult},
    tensor::{Shape, Tensor},
};
use cuda_rs::stream::CuStream;
use tensorrt_rs_sys::{
    runtime::{Runtime, CudaEngine, ExecutionContext},
    logger::Severity,
};
use std::{collections::HashMap, fs, path::Path};

pub struct TRTEngine {
    runtime: Option<Runtime>,
    engine: Option<CudaEngine>,
    context: Option<ExecutionContext>,
    stream: CuStream,
    tensors: HashMap<String, Tensor>,
}

impl TRTEngine {
    pub fn new<P: AsRef<Path>>(engine_path: &P, stream: &CuStream) -> TRTResult<Self> {
        let mut runtime = match Runtime::new() {
            Some(runtime) => runtime,
            None => return Err(TRTError::RuntimeCreationError),
        };

        let data = fs::read(engine_path)?;

        let engine = match runtime.deserialize(data.as_slice()) {
            Some(engine) => engine,
            None => return Err(TRTError::EngineDeserializationError),
        };

        Ok(Self {
            runtime: Some(runtime),
            engine: Some(engine),
            context: None,
            stream: stream.clone(),
            tensors: HashMap::new(),
        })
    }

    // TODO: reuse device memory
    pub fn activate(&mut self) -> TRTResult<()> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        self.context = match engine.create_execution_context() {
            Some(context) => Some(context),
            None => return Err(TRTError::ExecutionContextCreationError),
        };

        Ok(())
    }

    pub fn allocate_io_tensors(
        &mut self,
        max_shape_dict: &HashMap<&str, &Shape>,
        stream: Option<&CuStream>,
    ) -> TRTResult<()> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        let context: &mut ExecutionContext = match self.context.as_mut() {
            Some(context) => context,
            None => return Err(TRTError::ExecutionContextNotInitialized),
        };
        let stream = match stream {
            Some(stream) => stream,
            None => &self.stream,
        };

        let num_io_tensors = engine.get_num_io_tensors();

        for i in 0..num_io_tensors {
            let name = engine.get_io_tensor_name(i);
            let shape = engine.get_tensor_shape(name);
            let shape = Shape(shape);
            let shape = match max_shape_dict.get(name) {
                Some(max_shape) => max_shape,
                None => &shape,
            };
            if shape.0.iter().any(|&dim| dim < 0) {
                return Err(TRTError::ShapeError(shape.0.clone()));
            }
            if engine.get_tensor_io_mode(name).is_input() {
                if !context.set_input_shape(name, shape.0.as_slice()) {
                    return Err(TRTError::ShapeError(shape.0.clone()));
                }
            }

            let dtype = engine.get_tensor_dtype(name);
            let tensor = Tensor::empty(&shape, dtype, stream)?;
            let ptr = unsafe { tensor.get_raw_ptr() };
            self.tensors.insert(name.to_string(), tensor);
            if !context.set_tensor_address(name, ptr as _) {
                return Err(TRTError::InvalidAddress);
            }
        }

        // TODO: validate shapes, (batch size)

        Ok(())
    }

    // TODO: use cuda graph
    pub fn inference(
        &mut self,
        feed_dict: &HashMap<&str, &Tensor>,
        stream: Option<&CuStream>,
    ) -> TRTResult<&HashMap<String, Tensor>> {
        let context: &mut ExecutionContext = match self.context.as_mut() {
            Some(context) => context,
            None => return Err(TRTError::ExecutionContextNotInitialized),
        };
        let stream = match stream {
            Some(stream) => stream,
            None => &self.stream,
        };

        for (name, input_tensor) in feed_dict {
            let tensor = match self.tensors.get_mut(name.to_owned()) {
                Some(tensor) => tensor,
                None => continue,
            };
            let new_shape = input_tensor.shape();
            if tensor.shape() != new_shape {
                unsafe { tensor.reset_shape(new_shape)? };
                if !context.set_input_shape(name, new_shape.0.as_slice()) {
                    return Err(TRTError::ShapeError(new_shape.0.clone()));
                }
            }
            tensor.copy_from(input_tensor, Some(stream))?;
        }

        // TODO: validate shapes, (batch size)

        if !context.enqueue_v3(stream) {
            return Err(TRTError::EnqueueError);
        }

        Ok(&self.tensors)
    }

    pub fn log(&mut self, level: Severity, msg: &str) {
        self.runtime.as_mut().unwrap().logger().log(level, msg);
    }
}

impl Drop for TRTEngine {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            std::mem::drop(context);
        }

        if let Some(engine) = self.engine.take() {
            std::mem::drop(engine);
        }

        if let Some(runtime) = self.runtime.take() {
            std::mem::drop(runtime);
        }
    }
}
