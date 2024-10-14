fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_version, values(\"12\"))");
    if cudasys::types::cuda::CUDA_VERSION >= 12000 {
        println!("cargo::rustc-cfg=cuda_version=\"12\"");
    }

    println!("cargo:rerun-if-changed=build.rs");

    // TODO: use bindgen (or cuda_hook) to automatically generate the FFI
}
