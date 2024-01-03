use clap::Parser;
use cuda_rs::{device::CuDevice, stream::CuStream};
use tensorrt::{TRTEngine, TRTResult, Shape, DataType, Tensor};
use tch::vision::imagenet::load_image_and_resize224;
use std::{collections::HashMap, path::Path};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    engine_path: String,

    #[arg(short, long)]
    image_path: String,
}

fn main() -> TRTResult<()> {
    let args = Args::parse();
    let Args { engine_path, image_path } = args;
    let engine_path = Path::new(&engine_path);

    let image = load_image_and_resize224(&image_path).unwrap();
    let image = image.contiguous();
    let image = image.unsqueeze(0);
    let host_ptr = image.data_ptr();

    cuda_rs::init()?;

    let input_shape = Shape(vec![1, 3, 224, 224]);
    let output_shape = Shape(vec![1, 768]);
    let dtype = DataType::FLOAT;
    let mem_size = input_shape.size() * dtype.get_elem_size();

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
        ("images", &input_shape),
        ("features", &output_shape),
    ]);
    engine.allocate_io_tensors(&max_shape_dict, None)?;

    let feed_dict: HashMap<&str, &Tensor> = HashMap::from([
        ("images", &input_tensor),
    ]);
    let res = engine.inference(&feed_dict, None)?;

    stream.synchronize()?;

    let features = res.get("features").unwrap();
    let features_ptr = unsafe { features.get_raw_ptr() };

    let device = tch::Device::cuda_if_available();
    let features = unsafe {
        tch::Tensor::from_blob(
            features_ptr as _,
            vec![1, 768].as_slice(),
            vec![768, 1].as_slice(),
            tch::Kind::Float,
            device,
        )
    };
    let features = features.to_device(tch::Device::Cpu);

    let mean = features.mean(tch::Kind::Float);

    mean.print();

    Ok(())
}
