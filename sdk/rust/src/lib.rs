pub const SDK_VERSION: &str = "0.1.0-alpha.1";

pub fn sdk_version() -> &'static str {
    SDK_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!sdk_version().is_empty());
    }
}
