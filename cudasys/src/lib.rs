#![allow(non_snake_case, non_upper_case_globals, non_camel_case_types, warnings)]

pub mod cuda {
    include!("bindings/cuda.rs");
}

pub mod cudart {
    include!("bindings/cudart.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    // This should work without GPU
    #[test]
    fn get_version() {
        let mut version: i32 = 0;
        let result = unsafe { cudart::cudaDriverGetVersion(&mut version as *mut i32) };
        if result != cudart::cudaError::cudaSuccess {
            panic!("Cannot get driver version: ERROR={:?}", result);
        }
        println!("Version = {}", version);
    }
}
