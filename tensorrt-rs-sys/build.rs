use std::env;
use std::path::PathBuf;

fn find_dir(
    env_key: &'static str,
    candidates: Vec<&'static str>,
    file_to_find: &'static str,
) -> Option<PathBuf> {
    match env::var_os(env_key) {
        Some(val) => Some(PathBuf::from(&val)),
        _ => {
            for candidate in candidates {
                let path = PathBuf::from(candidate);
                let file_path = path.join(file_to_find);
                if file_path.exists() {
                    return Some(path);
                }
            }

            None
        }
    }
}

fn main() {
    let cuda_include_dir = find_dir(
        "CUDA_INCLUDE_PATH",
        vec!["/opt/cuda/include", "/usr/local/cuda/include"],
        "cuda.h",
    ).expect("Could not find CUDA include path");

    let tensorrt_include_dir = find_dir(
        "TENSORRT_INCLUDE_PATH",
        vec!["/usr/local/include", "/usr/include/x86_64-linux-gnu"],
        "NvInfer.h",
    ).expect("Could not find TensorRT include path");

    let tensorrt_library_dir = find_dir(
        "TENSORRT_LIBRARY_PATH",
        vec!["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"],
        "libnvinfer.so",
    ).expect("Could not find TensorRT library path");

    let include_files = vec![
        "cxx/include/logger.h",
        "cxx/include/runtime.h"
    ];
    let cpp_files = vec![
        "cxx/src/logger.cpp",
        "cxx/src/runtime.cpp"
    ];
    let rust_files = vec![
        "src/lib.rs",
    ];

    cxx_build::bridges(&rust_files)
        .include(cuda_include_dir)
        .include(tensorrt_include_dir)
        .include("cxx/include")
        .files(&cpp_files)
        .define("FMT_HEADER_ONLY", None)
        .flag_if_supported("-std=c++17")
        .compile("tensorrt-rs-sys-cxxbridge");

    println!("cargo:rustc-link-search={}", tensorrt_library_dir.to_string_lossy());

    let libraries = vec![
        "nvinfer",
        "nvinfer_plugin",
        "nvparsers",
    ];

    for library in libraries {
        println!("cargo:rustc-link-lib={}", library);
    }

    for file in include_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    for file in cpp_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    for file in rust_files {
        println!("cargo:rerun-if-changed={}", file);
    }
}
