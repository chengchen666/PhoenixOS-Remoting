fn main() {
    #[cfg(feature = "phos")]
    {
        let libpos_path =
            std::env::var("LIBPOS_PATH").expect("Cannot get libpos path. Please set LIBPOS_PATH");
        println!("cargo:rustc-link-search=native={libpos_path}");
    }

    println!("cargo:rerun-if-changed=build.rs");

    hookgen::generate_impls(
        "../cudasys/src/hooks/{}.rs",
        "../cudasys/src/bindings/funcs",
        "./src/dispatcher",
        "_exe",
        None,
        (cudasys::cuda::CUDA_VERSION / 1000) as u8,
    );
}
