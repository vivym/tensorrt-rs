#pragma once

#include <memory>
#include <NvInferRuntime.h>
#include "rust/cxx.h"
#include "logger.h"

namespace trt_rs::runtime {

using nvinfer1::IRuntime;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::Dims;
using logger::Logger;

class CudaEngine;

class Runtime {
public:
    Runtime(std::unique_ptr<IRuntime> runtime) : runtime_(std::move(runtime)) {}

    std::unique_ptr<CudaEngine> deserialize(rust::Slice<const std::uint8_t> data, std::size_t size) noexcept;

private:
    std::unique_ptr<IRuntime> runtime_;
};

class ExecutionContext;

class CudaEngine {
public:
    CudaEngine(std::unique_ptr<ICudaEngine> engine) : engine_(std::move(engine)) {}

    std::unique_ptr<ExecutionContext> create_execution_context() noexcept;

    int32_t get_num_io_tensors() const noexcept {
        return engine_->getNbIOTensors();
    }

    rust::Str get_io_tensor_name(int32_t index) const noexcept {
        return engine_->getIOTensorName(index);
    }

    int32_t get_tensor_io_mode(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return static_cast<int32_t>(engine_->getTensorIOMode(name_str.c_str()));
    }

    rust::Vec<int32_t> get_tensor_shape(rust::Str name) const noexcept;
private:
    std::unique_ptr<ICudaEngine> engine_;
};

class ExecutionContext {
public:
    ExecutionContext(std::unique_ptr<IExecutionContext> context) : context_(std::move(context)) {}

    bool set_input_shape(rust::Str name, rust::Slice<const int32_t> dims) noexcept;

    bool all_input_dimensions_specified() const noexcept {
        return context_->allInputDimensionsSpecified();
    }

    bool set_tensor_address(rust::Str name, std::size_t address) noexcept {
        const auto name_str = std::string(name);
        return context_->setTensorAddress(name_str.c_str(), reinterpret_cast<void*>(address));
    }

    bool enqueue_v3(std::size_t stream) noexcept;
private:
    std::unique_ptr<IExecutionContext> context_;
};

std::unique_ptr<Runtime> create_runtime(Logger& logger);

} // namespace trt_rs::runtime

