use crate::error::{TRTError, TRTResult};
use cuda_rs::{memory::DeviceMemory, stream::CuStream};
use tensorrt_rs_sys::runtime::DataType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<i32>);

impl Shape {
    pub fn size(&self) -> usize {
        self.0
            .iter()
            .map(|x| *x as usize)
            .product::<usize>()
    }
}

pub struct Tensor {
    mem: DeviceMemory,
    shape: Shape,
    dtype: DataType,
}

impl Tensor {
    pub fn empty(shape: &Shape, dtype: DataType, stream: &CuStream) -> TRTResult<Self> {
        let mem_size = shape.size() * dtype.get_elem_size();
        let mem = DeviceMemory::new(mem_size, stream)?;
        Ok(Self { mem, shape: shape.clone(), dtype })
    }

    pub fn from_memory(mem: DeviceMemory, shape: &Shape, dtype: DataType) -> Self {
        Self { mem, shape: shape.clone(), dtype }
    }

    pub fn get_memory(&self) -> &DeviceMemory {
        &self.mem
    }

    pub fn from_raw_ptr(
        ptr: usize, shape: &Shape, dtype: DataType, stream: &CuStream
    ) -> Self {
        let mem_size = shape.size() * dtype.get_elem_size();
        let mem = unsafe {
            DeviceMemory::from_raw(ptr as _, mem_size, stream)
        };
        Self { mem, shape: shape.clone(), dtype }
    }

    pub unsafe fn get_raw_ptr(&self) -> usize {
        self.mem.get_raw() as usize
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub unsafe fn reset_shape(&mut self, shape: &Shape) -> TRTResult<()> {
        if self.shape.size() < shape.size() {
            return Err(TRTError::ResetShapesError);
        }
        self.shape = shape.clone();
        Ok(())
    }

    pub fn copy_from(&mut self, src: &Self, stream: Option<&CuStream>) -> TRTResult<()> {
        if self.shape != src.shape {
            return Err(TRTError::ShapeMismatch);
        }
        if self.dtype != src.dtype {
            return Err(TRTError::DTypeMismatch);
        }
        self.mem.copy_from(&src.mem, stream)?;

        Ok(())
    }
}
