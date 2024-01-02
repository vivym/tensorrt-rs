#include "runtime.h"

namespace trt_rs::runtime {

std::unique_ptr<CudaEngine>
Runtime::deserialize(rust::Slice<const std::uint8_t> data) noexcept {
    auto engine = runtime_->deserializeCudaEngine(data.data(), data.size());
    if (!engine) {
        return nullptr;
    } else {
        return std::make_unique<CudaEngine>(std::unique_ptr<ICudaEngine>(engine));
    }
}

rust::Vec<int32_t> CudaEngine::get_tensor_shape(rust::Str name) const noexcept {
    const auto name_str = std::string(name);
    const auto dims = engine_->getTensorShape(name_str.c_str());
    auto dims_vec = rust::Vec<int32_t>();
    dims_vec.reserve(dims.nbDims);
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        dims_vec.push_back(dims.d[i]);
    }
    return dims_vec;
}

std::unique_ptr<ExecutionContext>
CudaEngine::create_execution_context() noexcept {
    auto context = engine_->createExecutionContext();
    if (!context) {
        return nullptr;
    } else {
        return std::make_unique<ExecutionContext>(std::unique_ptr<IExecutionContext>(context));
    }
}

std::unique_ptr<ExecutionContext>
CudaEngine::create_execution_context_without_device_memory() noexcept {
    auto context = engine_->createExecutionContextWithoutDeviceMemory();
    if (!context) {
        return nullptr;
    } else {
        return std::make_unique<ExecutionContext>(std::unique_ptr<IExecutionContext>(context));
    }
}

rust::Vec<int32_t> ExecutionContext::get_tensor_strides(rust::Str name) const noexcept {
    const auto name_str = std::string(name);
    const auto dims = context_->getTensorStrides(name_str.c_str());
    auto vec = rust::Vec<int32_t>();
    vec.reserve(dims.nbDims);
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        vec.push_back(dims.d[i]);
    }
    return vec;
}

bool ExecutionContext::set_input_shape(rust::Str name, rust::Slice<const int32_t> dims) noexcept {
    const auto name_str = std::string(name);
    const int32_t nb_dims = dims.size();
    Dims dims_trt;
    dims_trt.nbDims = nb_dims;
    for (int32_t i = 0; i < nb_dims; ++i) {
        dims_trt.d[i] = dims[i];
    }
    return context_->setInputShape(name_str.c_str(), dims_trt);
}

rust::Vec<int32_t> ExecutionContext::get_tensor_shape(rust::Str name) const noexcept {
    const auto name_str = std::string(name);
    const auto dims = context_->getTensorShape(name_str.c_str());
    auto dims_vec = rust::Vec<int32_t>();
    dims_vec.reserve(dims.nbDims);
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        dims_vec.push_back(dims.d[i]);
    }
    return dims_vec;
}

bool ExecutionContext::enqueue_v3(std::size_t stream) noexcept {
    return context_->enqueueV3(reinterpret_cast<cudaStream_t>(stream));
}

std::unique_ptr<Runtime> create_runtime(Logger& logger) {
    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        return nullptr;
    } else {
        return std::make_unique<Runtime>(std::unique_ptr<IRuntime>(runtime));
    }
}

} // namespace trt_rs::runtime
