#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "trt_rs::logger"]
    unsafe extern "C++" {
        include!("tensorrt-rs-sys/cxx/include/logger.h");

        type Logger;

        fn create_logger() -> UniquePtr<Logger>;

        fn log(self: Pin<&mut Logger>, severity: i32, msg: &str);

        fn set_level(self: Pin<&mut Logger>, severity: i32);
    }

    #[namespace = "trt_rs::runtime"]
    unsafe extern "C++" {
        include!("tensorrt-rs-sys/cxx/include/runtime.h");

        type Runtime;

        type CudaEngine;

        type ExecutionContext;

        // Runtime
        fn create_runtime(logger: Pin<&mut Logger>) -> UniquePtr<Runtime>;

        fn deserialize(self: Pin<&mut Runtime>, data: &[u8]) -> UniquePtr<CudaEngine>;

        fn set_max_threads(self: Pin<&mut Runtime>, max_threads: i32) -> bool;

        fn get_max_threads(self: &Runtime) -> i32;

        fn set_engine_host_code_allowed(self: Pin<&mut Runtime>, allowed: bool);

        fn get_engine_host_code_allowed(self: &Runtime) -> bool;

        // CudaEngine
        fn get_tensor_shape(self: &CudaEngine, name: &str) -> Vec<i32>;

        fn get_tensor_dtype(self: &CudaEngine, name: &str) -> i32;

        fn get_num_layers(self: &CudaEngine) -> i32;

        fn create_execution_context(self: Pin<&mut CudaEngine>) -> UniquePtr<ExecutionContext>;

        fn is_shape_inference_io(self: &CudaEngine, name: &str) -> bool;

        fn get_tensor_io_mode(self: &CudaEngine, name: &str) -> i32;

        fn create_execution_context_without_device_memory(self: Pin<&mut CudaEngine>) -> UniquePtr<ExecutionContext>;

        fn get_device_memory_size(self: &CudaEngine) -> usize;

        fn is_refittable(self: &CudaEngine) -> bool;

        fn get_tensor_bytes_per_component(self: &CudaEngine, name: &str) -> i32;

        fn get_tensor_components_per_element(self: &CudaEngine, name: &str) -> i32;

        fn get_tensor_format(self: &CudaEngine, name: &str) -> i32;

        fn get_tensor_vectorized_dim(self: &CudaEngine, name: &str) -> i32;

        fn get_name(self: &CudaEngine) -> &str;

        fn get_num_optimization_profiles(self: &CudaEngine) -> i32;

        fn get_engine_capability(self: &CudaEngine) -> i32;

        fn has_implicit_batch_dimension(self: &CudaEngine) -> bool;

        fn get_num_io_tensors(self: &CudaEngine) -> i32;

        fn get_io_tensor_name(self: &CudaEngine, index: i32) -> &str;

        fn get_hardware_compatibility_level(self: &CudaEngine) -> i32;

        fn get_num_aux_streams(self: &CudaEngine) -> i32;

        // ExecutionContext
        fn set_debug_sync(self: Pin<&mut ExecutionContext>, sync: bool);

        fn get_debug_sync(self: &ExecutionContext) -> bool;

        fn set_name(self: Pin<&mut ExecutionContext>, name: &str);

        fn get_name(self: &ExecutionContext) -> &str;

        fn set_device_memory(self: Pin<&mut ExecutionContext>, memory: usize);

        fn get_tensor_strides(self: &ExecutionContext, name: &str) -> Vec<i32>;

        fn get_optimization_profile(self: &ExecutionContext) -> i32;

        fn set_input_shape(self: Pin<&mut ExecutionContext>, name: &str, shape: &[i32]) -> bool;

        fn get_tensor_shape(self: &ExecutionContext, name: &str) -> Vec<i32>;

        fn all_input_dimensions_specified(self: &ExecutionContext) -> bool;

        fn all_input_shapes_specified(self: &ExecutionContext) -> bool;

        fn set_optimization_profile_async(self: Pin<&mut ExecutionContext>, profile_index: i32, stream: usize) -> bool;

        fn set_enqueue_emits_profile(self: Pin<&mut ExecutionContext>, emits: bool);

        fn get_enqueue_emits_profile(self: &ExecutionContext) -> bool;

        fn report_to_profiler(self: Pin<&mut ExecutionContext>);

        fn set_tensor_address(self: Pin<&mut ExecutionContext>, name: &str, address: usize) -> bool;

        fn get_tensor_address(self: &ExecutionContext, name: &str) -> usize;

        fn set_input_tensor_address(self: Pin<&mut ExecutionContext>, name: &str, address: usize) -> bool;

        fn get_output_tensor_address(self: &ExecutionContext, name: &str) -> usize;

        fn set_input_consumed_event(self: Pin<&mut ExecutionContext>, event: usize) -> bool;

        fn get_input_consumed_event(self: &ExecutionContext) -> usize;

        fn get_max_output_size(self: &ExecutionContext, name: &str) -> usize;

        fn enqueue_v3(self: Pin<&mut ExecutionContext>, stream: usize) -> bool;

        fn set_persistent_cache_limit(self: Pin<&mut ExecutionContext>, limit: usize);

        fn get_persistent_cache_limit(self: &ExecutionContext) -> usize;

        fn set_nvtx_verbosity(self: Pin<&mut ExecutionContext>, verbosity: i32);

        fn set_aux_streams(self: Pin<&mut ExecutionContext>, streams: &[usize]);
    }

    #[namespace = "trt_rs::plugin"]
    unsafe extern "C++" {
        include!("tensorrt-rs-sys/cxx/include/plugin.h");

        fn load_library(plugin_path: &str) -> usize;

        fn unload_library(handle: usize);
    }
}

pub mod logger;
pub mod plugin;
pub mod runtime;
