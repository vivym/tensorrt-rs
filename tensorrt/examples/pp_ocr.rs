use clap::Parser;
use cuda_rs::{device::CuDevice, stream::CuStream};
use tensorrt::{TRTEngine, TRTResult, Shape, DataType, Tensor};
use std::{collections::HashMap, path::Path};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    engine: String,

    #[arg(short, long)]
    input_image: String,
}

fn main() -> TRTResult<()> {
    let args = Args::parse();
    let Args { engine: engine_path, .. } = args;
    let engine_path = Path::new(&engine_path);

    // 1 * 3 * 352 * 640
    let input_shape = Shape(vec![1, 3, 352, 640]);
    let output_shape = Shape(vec![1, 1, 352, 640]);
    let dtype = DataType::FLOAT;

    let mem_size = input_shape.size() * dtype.get_elem_size();
    let mut host_data = vec![0.0f32; mem_size];
    let host_ptr = host_data.as_mut_ptr();

    cuda_rs::init()?;

    let device = CuDevice::new(0)?;
    let ctx = device.retain_primary_context()?;
    let _guard = ctx.guard()?;
    let stream = CuStream::new()?;

    let input_tensor = Tensor::empty(&input_shape, dtype, &stream)?;
    input_tensor.get_memory().copy_from_raw(
        host_ptr as _, mem_size, Some(&stream)
    )?;

    let mut engine = TRTEngine::new(&engine_path, &stream).unwrap();

    engine.activate()?;

    let max_shape_dict = HashMap::from([
        ("x", &input_shape),
        ("sigmoid_0.tmp_0", &output_shape),
    ]);
    engine.allocate_io_tensors(&max_shape_dict, None)?;

    let feed_dict = HashMap::from([
        ("x", &input_tensor),
    ]);
    engine.inference(&feed_dict, None)?;

    stream.synchronize()?;

    // TODO: post-processing

    println!("Done");

    Ok(())
}
