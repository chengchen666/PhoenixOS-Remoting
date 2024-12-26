use std::{env, path::PathBuf};

fn main() {
    create_cuda_symlinks();
    hookgen::generate_impls(
        "../cudasys/src/hooks/{}.rs",
        "../cudasys/src/bindings/funcs",
        "./src/hijack",
        "_hijack",
        Some("_unimplement"),
        (cudasys::cuda::CUDA_VERSION / 1000) as u8,
    );

    println!("cargo:rerun-if-changed=build.rs");
}

/// Some `dlopen()` calls escaped our hook in `dl.rs`.
/// This prevents them from loading local CUDA libs.
fn create_cuda_symlinks() {
    let mut symlink_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    while symlink_dir.file_name().unwrap().to_str().unwrap() != "build" {
        assert!(symlink_dir.pop());
    }
    assert!(symlink_dir.pop());
    symlink_dir.push("cuda-symlinks");
    let _ = std::fs::create_dir(&symlink_dir);
    for lib in [
        "libcuda.so.1",
        "libcudart.so.11.0",
        "libcudart.so.12",
        "libnvidia-ml.so.1",
        "libcudnn.so.8",
        "libcudnn.so.9",
        "libcublas.so.11",
        "libcublas.so.12",
        "libcublasLt.so.11",
        "libcublasLt.so.12",
        "libnvrtc.so.11.2",
        "libnvrtc.so.11.3",
    ] {
        let _ = std::os::unix::fs::symlink("../libclient.so", symlink_dir.join(lib));
    }
}
