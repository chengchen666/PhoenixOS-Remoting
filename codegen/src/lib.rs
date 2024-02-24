extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Ident, Lit, NestedMeta};

mod utils;
use utils::{Element, ElementMode};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for server usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate execution functions for the server dispatcher.
///
/// To use this macro, annotate a call to `gen_exe` with the desired function name as the first
/// string literal argument, followed by the types of the return and parameters as string literals.
///
/// ### Example
/// We have a function `cudaSetDevice` with the following signature:
///
/// ```ignore
/// pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// To use this macro generating a helper function for the server dispatcher, we can write:
///
/// ```ignore
/// gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
/// ```
///
/// This invocation generates a function `cudaSetDeviceExe` with `channel_sender` and `channel_receiver` for communication.
/// `cudaSetDeviceExe` will be called by the server dispatcher to execute the native `cudaSetDevice` function and send the result back to the client.
///
/// Specifically, the function is expanded as:
///
/// ```ignore
/// pub fn cudaSetDeviceExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
///     info!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
///
///     let mut param1: ::std::os::raw::c_int = Default::default();
///
///     channel_receiver.recv_var(&mut param1).unwrap();
///
///     let result = unsafe { cudaSetDevice(param1) };
///
///     channel_sender.send_var(&result).unwrap();
///     channel_sender.flush_out().unwrap();
/// }
/// ```
///
#[proc_macro]
pub fn gen_exe(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::AttributeArgs);

    let func = match &input[0] {
        NestedMeta::Lit(Lit::Str(lit_str)) => Ident::new(&lit_str.value(), lit_str.span()),
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let func_exe = Ident::new(&format!("{}Exe", func), func.span());

    let mut input = input.iter();

    let result = match input.nth(1).unwrap() {
        NestedMeta::Lit(Lit::Str(lit_str)) => {
            let ty = syn::parse_str::<syn::Type>(&lit_str.value()).expect("Expected valid type");
            Element {
                name: format_ident!("result"),
                ty: ty,
                mode: ElementMode::Output,
            }
        }
        _ => panic!("Expected type as a string literal"),
    };

    let params: Vec<Element> = input
        .enumerate()
        .map(|(i, arg)| {
            match arg {
                NestedMeta::Lit(Lit::Str(lit_str)) => {
                    // list_str can be: - "type", - "*mut type"
                    let mut ty_str = lit_str.value();
                    if ty_str.starts_with("*mut ") {
                        ty_str = ty_str.replace("*mut ", "");
                        let ty = syn::parse_str::<syn::Type>(&ty_str).expect("Expected valid type");
                        Element {
                            name: format_ident!("param{}", i + 1),
                            ty: ty,
                            mode: ElementMode::Output,
                        }
                    } else {
                        let ty = syn::parse_str::<syn::Type>(&ty_str).expect("Expected valid type");
                        Element {
                            name: format_ident!("param{}", i + 1),
                            ty: ty,
                            mode: ElementMode::Input,
                        }
                    }
                }
                _ => panic!("Expected type as a string literal"),
            }
        })
        .collect();

    // definition statements
    let def_statements = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // receive parameters
    let recv_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! { channel_receiver.recv_var(&mut #name).unwrap(); }
        });

    // execution statement
    let exec_statement = {
        let result_name = &result.name;
        let result_ty = &result.ty;
        let params = params.iter().map(|param| {
            let name = &param.name;
            match param.mode {
                ElementMode::Input => quote! { #name },
                ElementMode::Output => quote! { &mut #name },
            }
        });
        quote! { let #result_name: #result_ty = unsafe { #func(#(#params),*) }; }
    };

    // send result
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .map(|param| {
            let name = &param.name;
            quote! { channel_sender.send_var(&#name).unwrap(); }
        });

    let gen_fn = quote! {
        pub fn #func_exe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
            info!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            #( #def_statements )*
            #( #recv_statements )*
            #exec_statement
            #( #send_statements )*
            channel_sender.send_var(&result).unwrap();
            channel_sender.flush_out().unwrap();
        }
    };

    gen_fn.into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Resevation.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate a Rust function for serializing parameters to a buffer.
///
/// ### Example
/// To use this macro, annotate a call to `gen_serialize` with the desired function name as the first
/// string literal argument, followed by the types of the parameters as string literals.
///
/// gen_serialize!("my_function", "i32", "String");
///
/// This invocation generates a function `my_function` with two parameters: the first of type `i32` and
/// the second of type `String` (with a `buffer`` defined in `network`).
/// When `my_function` is called with values for these parameters, it serializes
/// each parameter's into the buffer.
///
/// Specifically, the function is defined as:
///
/// fn my_function(buf : &mut RawBuffer, a1: i32, a2: String) -> Result<(),BufferError> {
///    ...
/// }
///
#[proc_macro]
pub fn gen_serialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::AttributeArgs);

    let fn_name = match &input[0] {
        NestedMeta::Lit(Lit::Str(lit_str)) => Ident::new(&lit_str.value(), lit_str.span()),
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, arg)| {
            let ty = match arg {
                NestedMeta::Lit(Lit::Str(lit_str)) => {
                    lit_str.parse::<syn::Type>().expect("Expected valid type")
                }
                _ => panic!("Expected type as a string literal"),
            };
            let param_name = format_ident!("a{}", i + 1);
            quote! { #param_name: &#ty }
        })
        .collect();

    let param_names: Vec<_> = (1..=params.len())
        .map(|i| format_ident!("a{}", i))
        .collect();

    let serialize_statements = param_names.iter().map(|name| {
        //        quote! { println!("[{}] {}: {:?}", stringify!(#fn_name), stringify!(#name), #name); }
        quote! { unsafe { buf.append(#name)? }; }
    });

    let gen_fn = quote! {
        fn #fn_name(buf : &mut RawBuffer, #(#params),*) -> Result<(),BufferError> {
            #( #serialize_statements )*
            Ok(())
        }
    };

    gen_fn.into()
}

/// The code is similar to gen_serialize, but it generates a Rust function for deserializing parameters from a buffer.
#[proc_macro]
pub fn gen_deserialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::AttributeArgs);

    let fn_name = match &input[0] {
        NestedMeta::Lit(Lit::Str(lit_str)) => Ident::new(&lit_str.value(), lit_str.span()),
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, arg)| {
            let ty = match arg {
                NestedMeta::Lit(Lit::Str(lit_str)) => {
                    lit_str.parse::<syn::Type>().expect("Expected valid type")
                }
                _ => panic!("Expected type as a string literal"),
            };
            let param_name = format_ident!("a{}", i + 1);
            quote! { #param_name: &mut #ty }
        })
        .collect();

    let param_names: Vec<_> = (1..=params.len())
        .map(|i| format_ident!("a{}", i))
        .collect();

    let deserialize_statements = param_names.iter().map(|name| {
        quote! { unsafe { *(#name) = buf.extract()? }; }
    });

    let gen_fn = quote! {
        fn #fn_name(buf : &mut RawBuffer, #(#params),*) -> Result<(),BufferError> {
            #( #deserialize_statements )*
            Ok(())
        }
    };

    gen_fn.into()
}
