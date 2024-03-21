use std::env;
use std::path::PathBuf;

fn main() {
    // The bindgen::Builder is the main entry point to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("/usr/local/cuda-11.3/include/cuda_runtime.h")
        // Whitelist types, functions, and variables
        .allowlist_type("^cuda.*")
        .allowlist_type("^surfaceReference")
        .allowlist_type("^textureReference")
        .allowlist_var("^cuda.*")
        .allowlist_function("^cuda.*")
        // Set the default enum style to be more Rust-like
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        // Disable documentation comments from being generated
        .generate_comments(false)
        // Add derives
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        // Tell cargo to tell rustc to link the system bzip2
        // shared library.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $CARGO_MANIFEST_DIR/src/cuda_runtime.rs.
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = root.join("src/bindings");
    bindings
        .write_to_file(out_dir.join("cuda_runtime.rs"))
        .expect("Couldn't write bindings!");
}
