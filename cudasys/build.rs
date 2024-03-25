extern crate glob;
use glob::glob;
use std::{
    collections::VecDeque,
    env,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

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

fn decorate(file_path: PathBuf) {
    // Read the file.
    let file = File::open(&file_path).expect(format!("Failed to open file {:?}", file_path).as_str());
    let reader = BufReader::new(file);
    let mut cache: VecDeque<String> = VecDeque::new();

    let mut decorated_buf = String::new();
    let mut emit = |line: &str| {
        decorated_buf.push_str(line);
        decorated_buf.push_str("\n");
    };

    // Process in line level.
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.starts_with("#[derive(") {
            cache.push_back(line);
            // No emitting.
        } else if line.starts_with("pub enum") {
            // sanity check
            assert_eq!(1, cache.len());
            // #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
            // -> #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default, FromPrimitive, codegen::Transportable)]
            let mut prev = cache.pop_front().unwrap();
            prev = prev.replace(")]", ", Default, FromPrimitive, codegen::Transportable)]");
            emit(&prev);
            emit(&line);
            // #[default]
            emit("#[default]");
        } else if line.starts_with("pub struct") || line.starts_with("pub union") {
            // sanity check
            assert_eq!(1, cache.len());
            // #[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
            // -> #[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, codegen::Transportable)]
            let mut prev = cache.pop_front().unwrap();
            prev = prev.replace(")]", ", codegen::Transportable)]");
            // if not contains `Default`
            // if !prev.contains("Default") {
            //     prev = prev.replace(")]", ", codegen::ZeroDefault)]");
            // }
            emit(&prev);
            emit(&line);
        } else if line.starts_with("pub type") && line.contains('*') {
            let index = line.find('*').unwrap();
            // sanity check
            if 0 != cache.len() {
                panic!("Cache is not empty: {:?}", cache);
            }
            // = *[mut, const] Struct;
            // -> = usize;
            let line = line[..index].to_string() + "usize;";
            emit(&line);
        } else {
            // sanity check
            if 0 != cache.len() {
                panic!("Cache is not empty: {:?}", cache);
            }
            emit(&line);
        }
    }

    // Overwrite the original file with decorated contents.
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&file_path)
        .expect("Failed to open bindings for writing");
    file.write_all(decorated_buf.as_bytes())
        .expect("Failed to write modified content");
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
    let out_file = root.join("src/bindings").join(format!("{}.rs", output));
    bindings
        .write_to_file(out_file.clone())
        .expect("Couldn't write bindings!");

    // Format the generated bindings for our purposes.
    decorate(out_file);
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
        vec![
            "^CU.*",
            "^cuuint(32|64)_t",
            "^cudaError_enum",
            "^cu.*Complex$",
            "^cuda.*",
            "^libraryPropertyType.*",
        ],
        vec!["^CU.*"],
        vec!["^cu.*"],
    );
}
