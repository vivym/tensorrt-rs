#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "trt_rs::logger"]
    unsafe extern "C++" {
        include!("tensorrt-rs-sys/cxx/include/logger.h");

        type Logger;

        fn create_logger() -> UniquePtr<Logger>;
        fn log(self: Pin<&mut Logger>, severity: i32, msg: &str);
    }

    #[namespace = "trt_rs::runtime"]
    unsafe extern "C++" {
        include!("tensorrt-rs-sys/cxx/include/runtime.h");

        type Runtime;
        type CudaEngine;
        type ExecutionContext;

        fn create_runtime(logger: Pin<&mut Logger>) -> UniquePtr<Runtime>;

        fn deserialize(self: Pin<&mut Runtime>, data: &[u8], size: usize) -> UniquePtr<CudaEngine>;

        fn create_execution_context(self: Pin<&mut CudaEngine>) -> UniquePtr<ExecutionContext>;

        fn get_num_io_tensors(self: &CudaEngine) -> i32;

        fn get_io_tensor_name(self: &CudaEngine, index: i32) -> &str;

        fn get_tensor_io_mode(self: &CudaEngine, name: &str) -> i32;

        fn get_tensor_shape(self: &CudaEngine, name: &str) -> Vec<i32>;

        fn set_input_shape(self: Pin<&mut ExecutionContext>, name: &str, shape: &[i32]) -> bool;

        fn all_input_dimensions_specified(self: &ExecutionContext) -> bool;

        fn set_tensor_address(self: Pin<&mut ExecutionContext>, name: &str, address: usize) -> bool;

        fn enqueue_v3(self: Pin<&mut ExecutionContext>, stream: usize) -> bool;
    }
}

pub mod logger;
pub mod runtime;
