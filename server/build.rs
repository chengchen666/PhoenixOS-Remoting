fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_version, values(\"12\"))");
    if cudasys::types::cuda::CUDA_VERSION >= 12000 {
        println!("cargo::rustc-cfg=cuda_version=\"12\"");
    }

    println!("cargo:rerun-if-changed=build.rs");

    hookgen::generate_impls(
        "../cudasys/src/hooks/{}.rs",
        "../cudasys/src/bindings/funcs",
        "./src/dispatcher/{}_exe.rs",
        None,
        (cudasys::cuda::CUDA_VERSION / 1000) as u8,
    );
}
