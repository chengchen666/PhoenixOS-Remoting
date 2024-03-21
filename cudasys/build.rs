extern crate glob;
use glob::glob;
use std::{env, path::PathBuf};

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        let split_char = ":";
        path.split(split_char).map(|s| PathBuf::from(s)).collect()
    } else {
        vec![]
    }
}

// output (candidates, valid_path)
pub fn find_cuda() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut candidates = read_env();
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    for e in glob("/usr/local/cuda-*").unwrap() {
        if let Ok(path) = e {
            candidates.push(path)
        }
    }

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = PathBuf::from(base).join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
            continue;
        }
    }
    eprintln!("Found CUDA paths: {:?}", valid_paths);
    (candidates, valid_paths)
}

fn bind_gen(
    paths: &Vec<PathBuf>,
    library: &str,
    output: &str,
    allowlist_types: Vec<&str>,
    allowlist_vars: Vec<&str>,
    allowlist_funcs: Vec<&str>,
) {
    // find the library header path
    let mut header_path = None;
    for path in paths {
        let header = path.join(format!("include/{}.h", library));
        if header.is_file() {
            header_path = Some(header);
            break;
        }
    }
    // find in this directory
    if header_path.is_none() {
        let header = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
            .join(format!("include/{}.h", library));
        if header.is_file() {
            header_path = Some(header);
        }
    }
    let header_path = header_path.expect("Could not find CUDA header file");

    // The bindgen::Builder is the main entry point to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header(header_path.to_str().unwrap());

    // Whitelist types, functions, and variables
    for ty in allowlist_types {
        bindings = bindings.allowlist_type(ty);
    }
    for var in allowlist_vars {
        bindings = bindings.allowlist_var(var);
    }
    for func in allowlist_funcs {
        bindings = bindings.allowlist_function(func);
    }

    bindings = bindings
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
        .derive_ord(true);
        // TODO: add callbacks
        // // Allow configuring different kinds of types in different situations.
        // .parse_callbacks(Box::new(bindgen::CargoCallbacks))

    // Add include paths
    for path in paths {
        bindings = bindings.clang_arg(format!("-I{}/include", path.to_str().unwrap()));
    }

    let bindings = bindings
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the `$CARGO_MANIFEST_DIR/src/library.rs`.
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = root.join("src/bindings");
    bindings
        .write_to_file(out_dir.join(format!("{}.rs", output)))
        .expect("Couldn't write bindings!");
}

fn main() {
    let (cuda_paths, cuda_libs) = find_cuda();

    // Tell rustc to link the CUDA library.
    for path in &cuda_libs {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    // Use bindgen to automatically generate the FFI (in `src/bindings`).
    bind_gen(
        &cuda_paths,
        "cuda_runtime",
        "cudart",
        vec!["^cuda.*", "^surfaceReference", "^textureReference"],
        vec!["^cuda.*"],
        vec!["^cuda.*"],
    );

    bind_gen(
        &cuda_paths,
        "cuda_wrapper",
        "cuda",
        vec!["^CU.*", "^cuuint(32|64)_t", "^cudaError_enum", "^cu.*Complex$", "^cuda.*", "^libraryPropertyType.*"],
        vec!["^CU.*"],
        vec!["^cu.*"],
    );
}
