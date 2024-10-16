extern crate glob;
use glob::glob;
extern crate regex;
use regex::Regex;
extern crate syn;
use syn::{Signature, Type, parse_str};
use syn::__private::ToTokens;
use std::{
    collections::{VecDeque, HashMap},
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};
extern crate toml;
extern crate serde;
use serde::Deserialize;

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        let split_char = ":";
        path.split(split_char).map(PathBuf::from).collect()
    } else {
        vec![]
    }
}

// output (candidates, valid_path)
pub fn find_cuda() -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut candidates = read_env();
    candidates.push(PathBuf::from(".")); // bindgen wrapper headers
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from("/usr"));
    for e in glob("/usr/local/cuda-*").unwrap() {
        if let Ok(path) = e {
            candidates.push(path)
        }
    }

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = base.join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
        }
        // cudnn
        let cudnn_lib = base.join("include/x86_64-linux-gnu");
        if cudnn_lib.is_dir() {
            valid_paths.push(cudnn_lib);
        }
    }
    eprintln!("Found CUDA paths: {:?}", valid_paths);
    (candidates, valid_paths)
}

fn decorate(reader: &str) -> String {
    let mut cache: VecDeque<String> = VecDeque::new();

    let mut decorated_buf = String::with_capacity(reader.len() * 2);
    let mut emit = |line: &str| {
        decorated_buf.push_str(line);
        decorated_buf.push('\n');
    };

    // Process in line level.
    for line in reader.lines() {
        if line.starts_with("#[derive(") {
            cache.push_back(line.to_owned());
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
            emit(&prev);
            emit(&line);
        } else if line.starts_with("pub type") && line.contains('*') {
            let index = line.find('*').unwrap();
            // sanity check
            if !cache.is_empty() {
                panic!("Cache is not empty: {:?}", cache);
            }
            // = *[mut, const] Struct;
            // -> = usize;
            let line = line[..index].to_string() + "usize;";
            emit(&line);
        } else {
            // sanity check
            if !cache.is_empty() {
                panic!("Cache is not empty: {:?}", cache);
            }
            emit(&line);
        }
    }

    decorated_buf
}

/// Read the file, split it into two parts: one with the types and the other with the functions.
/// Remove the original file.
fn split(content: &str, types_file: &Path, funcs_file: &Path) {
    // regex to match `extern "C" { ... }`
    let re = Regex::new(r#"(?s)extern "C" \{.*?\}\n"#).unwrap();

    // Extract all blocks that match the regex
    let funcs: Vec<_> = re.find_iter(&content).map(|mat| mat.as_str()).collect();
    let types = re.replace_all(&content, "");

    // Write the types and functions to separate files.
    if let Some(parent) = types_file.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("Failed to create directory {parent:?}: {e}"));
    }
    let mut types_file = File::create(types_file).unwrap_or_else(|e| panic!("Failed to create file {types_file:?}: {e}"));
    writeln!(types_file, "{}", types).expect("Failed to write types");

    if let Some(parent) = funcs_file.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("Failed to create directory {parent:?}: {e}"));
    }
    let mut funcs_file = File::create(funcs_file).unwrap_or_else(|e| panic!("Failed to create file {funcs_file:?}: {e}"));

    for f in funcs {
        write!(funcs_file, "{f}").expect("Failed to write function");
    }
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct HookConfig {
    // Design for the future: user self-defined API hook.
    default: bool,
    client_hook: Option<String>,
    server_hook: Option<String>,
}

#[derive(Deserialize)]
struct UserHook {
    cuda: HashMap<String, HookConfig>,
    cudart: HashMap<String, HookConfig>,
    nvml: HashMap<String, HookConfig>,
    cudnn: HashMap<String, HookConfig>,
    cublas: HashMap<String, HookConfig>
}

fn write(content: &str, output: &str) {
    // Write to client crate

    let hook_path = Path::new("../client/hook.toml");
    let hook_content = std::fs::read_to_string(&hook_path).unwrap_or_else(|e| panic!("Failed to read file {hook_path:?}: {e}"));
    let user_hooks: UserHook = toml::from_str(&hook_content).expect("Failed to parse hook.toml file");
    let user_hook = match output {
        "cuda" => &user_hooks.cuda,
        "cudart" => &user_hooks.cudart,
        "nvml" => &user_hooks.nvml,
        "cudnn" => &user_hooks.cudnn,
        "cublas" => &user_hooks.cublas,
        &_ => todo!(),
    };

    let re = Regex::new(r#"(?s)extern "C" \{.*?\}"#).unwrap();
    let funcs: Vec<_> = re.find_iter(&content).map(|mat| mat.as_str()).collect();

    let header = format!("#![allow(non_snake_case)]\nuse super::*;\nuse cudasys::types::{}::*;", output);

    let unimplemented_file = PathBuf::from(format!("../client/src/hijack/{output}_unimplement.rs"));
    if let Some(parent) = unimplemented_file.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("Failed to create directory {parent:?}: {e}"));
    }
    let mut unimplemented_file = File::create(&unimplemented_file).unwrap_or_else(|e| panic!("Failed to create file {unimplemented_file:?}: {e}"));
    writeln!(unimplemented_file, "{}\n", header).expect("Failed to write header");

    for f in funcs {
        let sig = parse_sig(f);
        match user_hook.get(&sig.func_name) {
            Some(_config) => {
                // TODO: read user config, then default gen_hijack or use client_hook and server_hook
            }
            _ => write_macro(&unimplemented_file, "gen_unimplement", sig),
        };
    }
}

fn bind_gen(
    paths: &[PathBuf],
    library_header: &str,
    output: &str,
    allowlist_types: &[&str],
    allowlist_vars: &[&str],
    allowlist_funcs: &[&str],
    link_lib: &str,
) {
    println!("cargo:rustc-link-lib={link_lib}");

    // find the library header path
    let mut header_path = None;
    for path in paths {
        let mut header = path.clone();
        header.push("include");
        header.push(library_header);
        if header.is_file() {
            header_path = Some(header);
            break;
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

    let root = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Format the generated bindings for our purposes.
    let out_file = decorate(&bindings.to_string());

    // Split the file into two parts: one with the types and the other with the functions.
    let types_file = PathBuf::from(format!("{root}/src/bindings/types/{output}.rs"));
    let funcs_file = PathBuf::from(format!("{root}/src/bindings/funcs/{output}.rs"));
    split(&out_file, &types_file, &funcs_file);

    // write gen_xxx macro to client/src/hijack folder, with name output_xxx
    write(&out_file, output);
}

struct SigParser {
    func_name: String,
    result_ty: String,
    params_ty: Vec<String>,
}

fn write_macro(mut file: &File, macro_name: &str, sig: SigParser) {
    let mut params = String::new();
    params.push_str(format!("\"{}\"", sig.func_name).as_str());
    params.push_str(format!(", \"{}\"", sig.result_ty).as_str());
    for p in &sig.params_ty {
        params.push_str(format!(", \"{}\"", p).as_str());
    }
    let _ = file.write_all(format!("{}!({});\n", macro_name, params).as_bytes());
}

fn parse_sig(input: &str) -> SigParser {
    let re = Regex::new(r#"(?s)extern "C" \{(.*?)\}"#).unwrap();
    let input = match re.captures(input) {
        Some(code) => {
            match code.get(1) {
                Some(sig) => sig.as_str().trim(),
                None => input,
            }
        }
        None => input,
    };
    let input = input.trim_start_matches("pub ").trim_end_matches(";");

    let sig = parse_str::<Signature>(input).unwrap_or_else(|e| panic!("Failed to parse {input:?}: {e}"));

    let func_name = sig.ident.to_string();

    let result_ty = match &sig.output {
        syn::ReturnType::Type(_, pat_ty) => type_to_string(pat_ty),
        _ => String::from(""),
    };

    let mut params_ty = Vec::new();
    for param in sig.inputs.iter() {
        if let syn::FnArg::Typed(pat_ty) = param {
            let ty = &pat_ty.ty;
            params_ty.push(type_to_string(ty));
        }
    }

    SigParser {
        func_name,
        result_ty,
        params_ty,
    }
}

fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::Ptr(ty_ptr) => {
            let const_token = if ty_ptr.const_token.is_some() {
                "const "
            } else {
                ""
            };

            let mutability = if ty_ptr.mutability.is_some() {
                "mut "
            } else {
                ""
            };
            let inner_type = type_to_string(&ty_ptr.elem);
            format!("*{}{}{}", const_token, mutability, inner_type)
        }
        Type::Path(_) => {
            ty.into_token_stream().to_string().replace(" ", "")
        }
        _ => {
            panic!("Unimplemented type {}", ty.into_token_stream());
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=../client/hook.toml");

    let (cuda_paths, cuda_libs) = find_cuda();

    // Tell rustc to link the CUDA library.
    for path in &cuda_libs {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
    eprintln!("CUDA paths: {:?}", cuda_paths);
    eprintln!("CUDA libs: {:?}", cuda_libs);

    // Use bindgen to automatically generate the FFI (in `src/bindings`).
    bind_gen(
        &cuda_paths,
        "cudart_wrapper.hpp",
        "cudart",
        &["^cuda.*", "^surfaceReference", "^textureReference"],
        &["^cuda.*", "CUDART_VERSION"],
        &["^cuda.*", "__cuda[A-Za-z]+"],
        "dylib=cudart",
    );

    bind_gen(
        &cuda_paths,
        "cuda_wrapper.h",
        "cuda",
        &[
            "^CU.*",
            "^cuuint(32|64)_t",
            "^cudaError_enum",
            "^cu.*Complex$",
            "^cuda.*",
            "^libraryPropertyType.*",
        ],
        &["^CU.*"],
        &["^cu.*"],
        "dylib=cuda",
    );

    bind_gen(
        &cuda_paths,
        "nvml.h",
        "nvml",
        &[],
        &[],
        &[],
        "dylib=nvidia-ml",
    );

    bind_gen(
        &cuda_paths,
        "cudnn.h",
        "cudnn",
        &["^cudnn.*", "^CUDNN.*"],
        &["^CUDNN.*", "^cudnn.*"],
        &["^cudnn.*"],
        "dylib=cudnn",
    );

    bind_gen(
        &cuda_paths,
        "cublas.h",
        "cublas",
        &["^cublas.*", "^CUBLAS.*"],
        &["^CUBLAS.*", "^cublas.*"],
        &["^cublas.*"],
        "dylib=cublas",
    );
}
