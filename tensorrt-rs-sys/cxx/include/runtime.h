#pragma once

#include <memory>
#include <NvInferRuntime.h>
#include "rust/cxx.h"
#include "logger.h"
#include "plugin.h"

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

    std::unique_ptr<CudaEngine> deserialize(rust::Slice<const std::uint8_t> data) noexcept;

    bool set_max_threads(int32_t threads) noexcept {
        return runtime_->setMaxThreads(threads);
    }

    int32_t get_max_threads() const noexcept {
        return runtime_->getMaxThreads();
    }

    void set_engine_host_code_allowed(bool allowed) noexcept {
        runtime_->setEngineHostCodeAllowed(allowed);
    }

    bool get_engine_host_code_allowed() const noexcept {
        return runtime_->getEngineHostCodeAllowed();
    }
private:
    std::unique_ptr<IRuntime> runtime_;
};

class ExecutionContext;

class CudaEngine {
public:
    CudaEngine(std::unique_ptr<ICudaEngine> engine) : engine_(std::move(engine)) {}

    rust::Vec<int32_t> get_tensor_shape(rust::Str name) const noexcept;

    int32_t get_tensor_dtype(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return static_cast<int32_t>(engine_->getTensorDataType(name_str.c_str()));
    }

    int32_t get_num_layers() const noexcept {
        return engine_->getNbLayers();
    }

    std::unique_ptr<ExecutionContext> create_execution_context() noexcept;

    bool is_shape_inference_io(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return engine_->isShapeInferenceIO(name_str.c_str());
    }

    int32_t get_tensor_io_mode(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return static_cast<int32_t>(engine_->getTensorIOMode(name_str.c_str()));
    }

    std::unique_ptr<ExecutionContext> create_execution_context_without_device_memory() noexcept;

    size_t get_device_memory_size() const noexcept {
        return engine_->getDeviceMemorySize();
    }

    bool is_refittable() const noexcept {
        return engine_->isRefittable();
    }

    int32_t get_tensor_bytes_per_component(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return engine_->getTensorBytesPerComponent(name_str.c_str());
    }

    int32_t get_tensor_components_per_element(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return engine_->getTensorComponentsPerElement(name_str.c_str());
    }

    int32_t get_tensor_format(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return static_cast<int32_t>(engine_->getTensorFormat(name_str.c_str()));
    }

    int32_t get_tensor_vectorized_dim(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return engine_->getTensorVectorizedDim(name_str.c_str());
    }

    rust::Str get_name() const noexcept {
        return engine_->getName();
    }

    int32_t get_num_optimization_profiles() const noexcept {
        return engine_->getNbOptimizationProfiles();
    }

    int32_t get_engine_capability() const noexcept {
        return static_cast<int32_t>(engine_->getEngineCapability());
    }

    bool has_implicit_batch_dimension() const noexcept {
        return engine_->hasImplicitBatchDimension();
    }

    int32_t get_num_io_tensors() const noexcept {
        return engine_->getNbIOTensors();
    }

    rust::Str get_io_tensor_name(int32_t index) const noexcept {
        return engine_->getIOTensorName(index);
    }

    int32_t get_hardware_compatibility_level() const noexcept {
        return static_cast<int32_t>(engine_->getHardwareCompatibilityLevel());
    }

    int32_t get_num_aux_streams() const noexcept {
        return engine_->getNbAuxStreams();
    }
private:
    std::unique_ptr<ICudaEngine> engine_;
};

class ExecutionContext {
public:
    ExecutionContext(std::unique_ptr<IExecutionContext> context) : context_(std::move(context)) {}

    void set_debug_sync(bool sync) noexcept {
        context_->setDebugSync(sync);
    }

    bool get_debug_sync() const noexcept {
        return context_->getDebugSync();
    }

    void set_name(rust::Str name) noexcept {
        const auto name_str = std::string(name);
        context_->setName(name_str.c_str());
    }

    rust::Str get_name() const noexcept {
        return context_->getName();
    }

    void set_device_memory(std::size_t memory) noexcept {
        context_->setDeviceMemory(reinterpret_cast<void*>(memory));
    }

    rust::Vec<int32_t> get_tensor_strides(rust::Str name) const noexcept;

    int32_t get_optimization_profile() const noexcept {
        return context_->getOptimizationProfile();
    }

    bool set_input_shape(rust::Str name, rust::Slice<const int32_t> dims) noexcept;

    rust::Vec<int32_t> get_tensor_shape(rust::Str name) const noexcept;

    bool all_input_dimensions_specified() const noexcept {
        return context_->allInputDimensionsSpecified();
    }

    bool all_input_shapes_specified() const noexcept {
        return context_->allInputShapesSpecified();
    }

    bool set_optimization_profile_async(int32_t profile, std::size_t stream) noexcept {
        return context_->setOptimizationProfileAsync(profile, reinterpret_cast<cudaStream_t>(stream));
    }

    void set_enqueue_emits_profile(bool emits) noexcept {
        context_->setEnqueueEmitsProfile(emits);
    }

    bool get_enqueue_emits_profile() const noexcept {
        return context_->getEnqueueEmitsProfile();
    }

    void report_to_profiler() noexcept {
        context_->reportToProfiler();
    }

    bool set_tensor_address(rust::Str name, std::size_t address) noexcept {
        const auto name_str = std::string(name);
        return context_->setTensorAddress(name_str.c_str(), reinterpret_cast<void*>(address));
    }

    std::size_t get_tensor_address(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return reinterpret_cast<std::size_t>(context_->getTensorAddress(name_str.c_str()));
    }

    bool set_input_tensor_address(rust::Str name, std::size_t address) noexcept {
        const auto name_str = std::string(name);
        return context_->setInputTensorAddress(name_str.c_str(), reinterpret_cast<void*>(address));
    }

    std::size_t get_output_tensor_address(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return reinterpret_cast<std::size_t>(context_->getOutputTensorAddress(name_str.c_str()));
    }

    // TODO: inferShapes

    bool set_input_consumed_event(std::size_t event) noexcept {
        return context_->setInputConsumedEvent(reinterpret_cast<cudaEvent_t>(event));
    }

    std::size_t get_input_consumed_event() const noexcept {
        return reinterpret_cast<std::size_t>(context_->getInputConsumedEvent());
    }

    // TODO: setOutputAllocator

    std::size_t get_max_output_size(rust::Str name) const noexcept {
        const auto name_str = std::string(name);
        return context_->getMaxOutputSize(name_str.c_str());
    }

    // TODO: setTemporaryStorageAllocator

    bool enqueue_v3(std::size_t stream) noexcept;

    void set_persistent_cache_limit(std::size_t limit) noexcept {
        context_->setPersistentCacheLimit(limit);
    }

    std::size_t get_persistent_cache_limit() const noexcept {
        return context_->getPersistentCacheLimit();
    }

    void set_nvtx_verbosity(int32_t verbosity) noexcept {
        context_->setNvtxVerbosity(static_cast<nvinfer1::ProfilingVerbosity>(verbosity));
    }

    void set_aux_streams(rust::Slice<const std::size_t> streams) noexcept {
        auto streams_ptr = const_cast<cudaStream_t*>(
            reinterpret_cast<cudaStream_t const*>(streams.data()));
        context_->setAuxStreams(streams_ptr, streams.size());
    }
private:
    std::unique_ptr<IExecutionContext> context_;
};

std::unique_ptr<Runtime> create_runtime(Logger& logger);

} // namespace trt_rs::runtime

