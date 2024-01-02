use crate::{ffi, logger::Logger};
use cxx::UniquePtr;
use cuda_rs::{event::CuEvent, stream::CuStream};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TensorIOMode {
    // Tensor is not an input or output.
    NONE = 0,
    // Tensor is input to the engine.
    INPUT = 1,
    // Tensor is output by the engine.
    OUTPUT = 2
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TensorFormat {
    // Row major linear format.
    // For a tensor with dimensions {N, C, H, W} or {numbers, channels,
    // columns, rows}, the dimensional index corresponds to {3, 2, 1, 0}
    // and thus the order is W minor.
    //
    // For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
    //
    LINEAR = 0,

    // Two wide channel vectorized row major format. This format is bound to
    // FP16. It is only available for dimensions >= 3.
    // For a tensor with dimensions {N, C, H, W},
    // the memory layout is equivalent to a C array with dimensions
    // [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][c/2][h][w][c%2].
    CHW2 = 1,

    // Eight channel format where C is padded to a multiple of 8. This format
    // is bound to FP16. It is only available for dimensions >= 3.
    // For a tensor with dimensions {N, C, H, W},
    // the memory layout is equivalent to the array with dimensions
    // [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][h][w][c].
    HWC8 = 2,

    // Four wide channel vectorized row major format. This format is bound to
    // INT8 or FP16. It is only available for dimensions >= 3.
    // For INT8, the C dimension must be a build-time constant.
    // For a tensor with dimensions {N, C, H, W},
    // the memory layout is equivalent to a C array with dimensions
    // [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][c/4][h][w][c%4].
    //
    // Deprecated usage:
    //
    // If running on the DLA, this format can be used for acceleration
    // with the caveat that C must be equal or lesser than 4.
    // If used as DLA input and the build option kGPU_FALLBACK is not specified,
    // it needs to meet line stride requirement of DLA format. Column stride in bytes should
    // be a multiple of 32 on Xavier and 64 on Orin.
    CHW4 = 3,

    // Sixteen wide channel vectorized row major format. This format is bound
    // to FP16. It is only available for dimensions >= 3.
    // For a tensor with dimensions {N, C, H, W},
    // the memory layout is equivalent to a C array with dimensions
    // [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][c/16][h][w][c%16].
    //
    // For DLA usage, this format maps to the native feature format for FP16,
    // and the tensor sizes are limited to C,H,W in the range [1,8192].
    //
    CHW16 = 4,

    // Thirty-two wide channel vectorized row major format. This format is
    // only available for dimensions >= 3.
    // For a tensor with dimensions {N, C, H, W},
    // the memory layout is equivalent to a C array with dimensions
    // [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][c/32][h][w][c%32].
    //
    // For DLA usage, this format maps to the native feature format for INT8,
    // and the tensor sizes are limited to C,H,W in the range [1,8192].
    CHW32 = 5,

    // Eight channel format where C is padded to a multiple of 8. This format
    // is bound to FP16, and it is only available for dimensions >= 4.
    // For a tensor with dimensions {N, C, D, H, W},
    // the memory layout is equivalent to an array with dimensions
    // [N][D][H][W][(C+7)/8*8], with the tensor coordinates (n, c, d, h, w)
    // mapping to array subscript [n][d][h][w][c].
    DHWC8 = 6,

    // Thirty-two wide channel vectorized row major format. This format is
    // bound to FP16 and INT8 and is only available for dimensions >= 4.
    // For a tensor with dimensions {N, C, D, H, W},
    // the memory layout is equivalent to a C array with dimensions
    // [N][(C+31)/32][D][H][W][32], with the tensor coordinates (n, c, d, h, w)
    // mapping to array subscript [n][c/32][d][h][w][c%32].
    CDHW32 = 7,

    // Non-vectorized channel-last format. This format is bound to either FP32 or UINT8,
    // and is only available for dimensions >= 3.
    HWC = 8,

    // DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
    // always has unit stride. The stride for stepping along the H axis is
    // rounded up to 64 bytes.
    //
    // The memory layout is equivalent to a C array with dimensions
    // [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
    // 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][c][h][w].
    DLALINEAR = 9,

    // DLA image format. For a tensor with dimension {N, C, H, W} the C axis
    // always has unit stride. The stride for stepping along the H axis is rounded up
    // to 32 bytes on Xavier and 64 bytes on Orin. C can only be 1, 3 or 4.
    // If C == 1, it will map to grayscale format.
    // If C == 3 or C == 4, it will map to color image format. And if C == 3,
    // the stride for stepping along the W axis needs to be padded to 4 in elements.
    //
    // When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
    // the memory layout is equivalent to a C array with dimensions
    // [N][H][roundUp(W, 32/C'/elementSize)][C'] on Xavier and [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
    // where elementSize is 2 for FP16
    // and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
    // subscript [n][h][w][c].
    DLAHWC4 = 10,

    // Sixteen channel format where C is padded to a multiple of 16. This format
    // is bound to FP16. It is only available for dimensions >= 3.
    // For a tensor with dimensions {N, C, H, W},
    // the memory layout is equivalent to the array with dimensions
    // [N][H][W][(C+15)/16*16], with the tensor coordinates (n, c, h, w)
    // mapping to array subscript [n][h][w][c].
    HWC16 = 11,

    // Non-vectorized channel-last format. This format is bound to FP32.
    // It is only available for dimensions >= 4.
    DHWC = 12
}

pub enum EngineCapability {
    //
    // Standard: TensorRT flow without targeting the safety runtime.
    // This flow supports both DeviceType::kGPU and DeviceType::kDLA.
    //
    STANDARD = 0,

    //
    // Safety: TensorRT flow with restrictions targeting the safety runtime.
    // See safety documentation for list of supported layers and formats.
    // This flow supports only DeviceType::kGPU.
    //
    // This flag is only supported in NVIDIA Drive(R) products.
    SAFETY = 1,

    //
    // DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes.
    // See DLA documentation for list of supported layers and formats.
    // This flow supports only DeviceType::kDLA.
    //
    DLASTANDALONE = 2,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum HardwareCompatibilityLevel {
    // Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was
    // built.
    NONE = 0,

    // Require that the engine is compatible with Ampere and newer GPUs. This will limit the max shared memory usage to
    // 48KiB, may reduce the number of available tactics for each layer, and may prevent some fusions from occurring.
    // Thus this can decrease the performance, especially for tf32 models.
    // This option will disable cuDNN, cuBLAS, and cuBLAS LT as tactic sources.
    AMPEREPLUS = 1,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ProfilingVerbosity {
    LAYERNAMESONLY = 0,     //< Print only the layer names. This is the default setting.
    NONE = 1,               //< Do not print any layer information.
    DETAILED = 2,           //< Print detailed layer information including layer names and layer parameters.
}

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

    pub fn set_max_threads(&mut self, max_threads: i32) -> bool {
        self.runtime.pin_mut().set_max_threads(max_threads)
    }

    pub fn get_max_threads(&self) -> i32 {
        self.runtime.get_max_threads()
    }

    pub fn set_engine_host_code_allowed(&mut self, allowed: bool) {
        self.runtime.pin_mut().set_engine_host_code_allowed(allowed)
    }

    pub fn get_engine_host_code_allowed(&self) -> bool {
        self.runtime.get_engine_host_code_allowed()
    }
}

pub struct CudaEngine(pub(crate) UniquePtr<ffi::CudaEngine>);

impl CudaEngine {
    pub fn get_tensor_shape(&self, name: &str) -> Vec<i32> {
        self.0.get_tensor_shape(name)
    }

    pub fn get_num_layers(&self) -> i32 {
        self.0.get_num_layers()
    }

    pub fn create_execution_context(&mut self) -> Option<ExecutionContext> {
        let context = self.0.pin_mut().create_execution_context();
        if context.is_null() {
            None
        } else {
            Some(ExecutionContext(context))
        }
    }

    pub fn is_shape_inference_io(&self, name: &str) -> bool {
        self.0.is_shape_inference_io(name)
    }

    pub fn get_tensor_io_mode(&self, name: &str) -> TensorIOMode {
        match self.0.get_tensor_io_mode(name) {
            0 => TensorIOMode::NONE,
            1 => TensorIOMode::INPUT,
            2 => TensorIOMode::OUTPUT,
            mode => panic!("Invalid tensor io mode: {}", mode),
        }
    }

    pub fn create_execution_context_without_device_memory(&mut self) -> Option<ExecutionContext> {
        let context =
            self.0.pin_mut().create_execution_context_without_device_memory();
        if context.is_null() {
            None
        } else {
            Some(ExecutionContext(context))
        }
    }

    pub fn get_device_memory_size(&self) -> usize {
        self.0.get_device_memory_size()
    }

    pub fn is_refittable(&self) -> bool {
        self.0.is_refittable()
    }

    pub fn get_tensor_bytes_per_component(&self, name: &str) -> i32 {
        self.0.get_tensor_bytes_per_component(name)
    }

    pub fn get_tensor_components_per_element(&self, name: &str) -> i32 {
        self.0.get_tensor_components_per_element(name)
    }

    pub fn get_tensor_format(&self, name: &str) -> TensorFormat {
        match self.0.get_tensor_format(name) {
            0 => TensorFormat::LINEAR,
            1 => TensorFormat::CHW2,
            2 => TensorFormat::HWC8,
            3 => TensorFormat::CHW4,
            4 => TensorFormat::CHW16,
            5 => TensorFormat::CHW32,
            6 => TensorFormat::DHWC8,
            7 => TensorFormat::CDHW32,
            8 => TensorFormat::HWC,
            9 => TensorFormat::DLALINEAR,
            10 => TensorFormat::DLAHWC4,
            11 => TensorFormat::HWC16,
            12 => TensorFormat::DHWC,
            format => panic!("Invalid tensor format: {}", format),
        }
    }

    pub fn get_tensor_vectorized_dim(&self, name: &str) -> i32 {
        self.0.get_tensor_vectorized_dim(name)
    }

    pub fn get_name(&self) -> &str {
        self.0.get_name()
    }

    pub fn get_num_optimization_profiles(&self) -> i32 {
        self.0.get_num_optimization_profiles()
    }

    pub fn get_engine_capability(&self) -> EngineCapability {
        match self.0.get_engine_capability() {
            0 => EngineCapability::STANDARD,
            1 => EngineCapability::SAFETY,
            2 => EngineCapability::DLASTANDALONE,
            capability => panic!("Invalid engine capability: {}", capability),
        }
    }

    pub fn has_implicit_batch_dimension(&self) -> bool {
        self.0.has_implicit_batch_dimension()
    }

    pub fn get_num_io_tensors(&self) -> i32 {
        self.0.get_num_io_tensors()
    }

    pub fn get_io_tensor_name(&self, index: i32) -> &str {
        self.0.get_io_tensor_name(index)
    }

    pub fn get_hardware_compatibility_level(&self) -> HardwareCompatibilityLevel {
        match self.0.get_hardware_compatibility_level() {
            0 => HardwareCompatibilityLevel::NONE,
            1 => HardwareCompatibilityLevel::AMPEREPLUS,
            level => panic!("Invalid hardware compatibility level: {}", level),
        }
    }

    pub fn get_num_aux_streams(&self) -> i32 {
        self.0.get_num_aux_streams()
    }
}

pub struct ExecutionContext(pub(crate) UniquePtr<ffi::ExecutionContext>);

impl ExecutionContext {
    pub fn set_debug_sync(&mut self, sync: bool) {
        self.0.pin_mut().set_debug_sync(sync)
    }

    pub fn get_debug_sync(&self) -> bool {
        self.0.get_debug_sync()
    }

    pub fn set_name(&mut self, name: &str) {
        self.0.pin_mut().set_name(name)
    }

    pub fn get_name(&self) -> &str {
        self.0.get_name()
    }

    pub fn set_device_memory(&mut self, memory: usize) {
        self.0.pin_mut().set_device_memory(memory)
    }

    pub fn get_tensor_strides(&self, name: &str) -> Vec<i32> {
        self.0.get_tensor_strides(name)
    }

    pub fn get_optimization_profile(&self) -> i32 {
        self.0.get_optimization_profile()
    }

    pub fn set_input_shape(&mut self, name: &str, shape: &[i32]) -> bool {
        self.0.pin_mut().set_input_shape(name, shape)
    }

    pub fn get_tensor_shape(&self, name: &str) -> Vec<i32> {
        self.0.get_tensor_shape(name)
    }

    pub fn all_input_dimensions_specified(&self) -> bool {
        self.0.all_input_dimensions_specified()
    }

    pub fn all_input_shapes_specified(&self) -> bool {
        self.0.all_input_shapes_specified()
    }

    pub fn set_optimization_profile_async(
        &mut self,
        profile_index: i32,
        stream: &CuStream,
    ) -> bool {
        let stream_raw = unsafe { stream.get_raw() };
        self.0
            .pin_mut()
            .set_optimization_profile_async(profile_index, stream_raw as _)
    }

    pub fn set_enqueue_emits_profile(&mut self, emits: bool) {
        self.0.pin_mut().set_enqueue_emits_profile(emits)
    }

    pub fn get_enqueue_emits_profile(&self) -> bool {
        self.0.get_enqueue_emits_profile()
    }

    pub fn report_to_profiler(&mut self) {
        self.0.pin_mut().report_to_profiler()
    }

    pub fn set_tensor_address(&mut self, name: &str, address: usize) -> bool {
        self.0.pin_mut().set_tensor_address(name, address)
    }

    pub fn get_tensor_address(&self, name: &str) -> usize {
        self.0.get_tensor_address(name)
    }

    pub fn set_input_tensor_address(&mut self, name: &str, address: usize) -> bool {
        self.0.pin_mut().set_input_tensor_address(name, address)
    }

    pub fn get_output_tensor_address(&self, name: &str) -> usize {
        self.0.get_output_tensor_address(name)
    }

    pub fn set_input_consumed_event(&mut self, event: &CuEvent) -> bool {
        let event_raw = unsafe { event.get_raw() };
        self.0.pin_mut().set_input_consumed_event(event_raw as _)
    }

    pub fn get_input_consumed_event(&self) -> CuEvent {
        let event_raw = self.0.get_input_consumed_event();
        unsafe { CuEvent::from_raw(event_raw as _) }
    }

    pub fn get_max_output_size(&self, name: &str) -> usize {
        self.0.get_max_output_size(name)
    }

    pub fn enqueue_v3(&mut self, stream: &CuStream) -> bool {
        let stream_raw = unsafe { stream.get_raw() };
        self.0.pin_mut().enqueue_v3(stream_raw as usize)
    }

    pub fn set_persistent_cache_limit(&mut self, limit: usize) {
        self.0.pin_mut().set_persistent_cache_limit(limit)
    }

    pub fn get_persistent_cache_limit(&self) -> usize {
        self.0.get_persistent_cache_limit()
    }

    pub fn set_nvtx_verbosity(&mut self, verbosity: ProfilingVerbosity) {
        self.0.pin_mut().set_nvtx_verbosity(verbosity as _)
    }

    pub fn set_aux_streams(&mut self, streams: &[&CuStream]) {
        let streams: Vec<_> = streams
            .iter()
            .map(|stream| unsafe { stream.get_raw() as _ })
            .collect();
        self.0.pin_mut().set_aux_streams(streams.as_slice())
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
