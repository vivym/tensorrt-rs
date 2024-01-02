use crate::ffi;
use cxx::UniquePtr;

pub struct Logger(pub(crate) UniquePtr<ffi::Logger>);

impl Logger {
    pub fn new() -> Self {
        Self(ffi::create_logger())
    }

    pub fn log(&mut self, severity: i32, msg: &str) {
        self.0.pin_mut().log(severity, msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger() {
        let mut logger = Logger::new();
        logger.log(0, "Hello, world!");
        logger.log(0, "Hello, world!");
        logger.log(0, "Hello, world!");
    }
}
