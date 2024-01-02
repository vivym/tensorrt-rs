use crate::ffi;
use cxx::UniquePtr;

pub struct Logger(pub(crate) UniquePtr<ffi::Logger>);

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Severity {
    InternalError = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Verbose = 4,
}

impl Logger {
    pub fn new() -> Self {
        Self(ffi::create_logger())
    }

    pub fn log(&mut self, severity: Severity, msg: &str) {
        self.0.pin_mut().log(severity as _, msg);
    }

    pub fn set_level(&mut self, severity: Severity) {
        self.0.pin_mut().set_level(severity as _);
    }

    pub fn error(&mut self, msg: &str) {
        self.log(Severity::Error, msg);
    }

    pub fn warning(&mut self, msg: &str) {
        self.log(Severity::Warning, msg);
    }

    pub fn info(&mut self, msg: &str) {
        self.log(Severity::Info, msg);
    }

    pub fn verbose(&mut self, msg: &str) {
        self.log(Severity::Verbose, msg);
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger() {
        let mut logger = Logger::new();
        logger.set_level(Severity::Info);
        logger.verbose("Hello, world!");
        logger.info("Hello, world!");
        logger.warning("Hello, world!");
        logger.error("Hello, world!");
        logger.log(Severity::InternalError, "Hello, world!");
    }
}
