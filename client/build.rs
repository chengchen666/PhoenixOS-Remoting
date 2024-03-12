use std::{env, path::PathBuf};

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let library_dir = root.join("src/elf");
    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=static=elf");
    println!("cargo:rerun-if-changed=build.rs");
}
