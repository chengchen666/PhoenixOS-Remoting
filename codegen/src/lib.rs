#![allow(non_snake_case)]

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Ident, Lit, NestedMeta};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for server usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate execution functions for the server dispatcher.
///
/// ### Example
/// To use this macro, annotate a call to `gen_exe` with the desired function name as the first
/// string literal argument, followed by the types of the parameters as string literals.
///
/// gen_exe!("myFunction", "i32", "String");
///
/// This invocation generates a function `myFunctionExe` with two parameters: the first of type `i32` and
/// the second of type `String` (with a `buffer`` defined in `network`). 
/// When `myFunction` is called with values for these parameters, it serializes
/// each parameter's into the buffer. 
///
/// Specifically, the function is defined as:
///
/// fn myFunctionExe(buf : &mut RawBuffer, a1: i32, a2: String) -> Result<(),BufferError> {
///    ... 
/// }
/// 
#[proc_macro]
pub fn gen_exe(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::AttributeArgs);

    let fn_name = match &input[0] {
        NestedMeta::Lit(Lit::Str(lit_str)) => Ident::new(&lit_str.value(), lit_str.span()),
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input.iter().skip(1).enumerate().map(|(i, arg)| {
        let ty = match arg {
            NestedMeta::Lit(Lit::Str(lit_str)) => lit_str.parse::<syn::Type>().expect("Expected valid type"),
            _ => panic!("Expected type as a string literal"),
        };
        let param_name = format_ident!("a{}", i + 1);
        quote! { #param_name: &#ty }
    }).collect();

    let param_names: Vec<_> = (1..=params.len()).map(|i| format_ident!("a{}", i)).collect();

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

    let params: Vec<_> = input.iter().skip(1).enumerate().map(|(i, arg)| {
        let ty = match arg {
            NestedMeta::Lit(Lit::Str(lit_str)) => lit_str.parse::<syn::Type>().expect("Expected valid type"),
            _ => panic!("Expected type as a string literal"),
        };
        let param_name = format_ident!("a{}", i + 1);
        quote! { #param_name: &#ty }
    }).collect();

    let param_names: Vec<_> = (1..=params.len()).map(|i| format_ident!("a{}", i)).collect();

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

    let params: Vec<_> = input.iter().skip(1).enumerate().map(|(i, arg)| {
        let ty = match arg {
            NestedMeta::Lit(Lit::Str(lit_str)) => lit_str.parse::<syn::Type>().expect("Expected valid type"),
            _ => panic!("Expected type as a string literal"),
        };
        let param_name = format_ident!("a{}", i + 1);
        quote! { #param_name: &mut #ty }
    }).collect();

    let param_names: Vec<_> = (1..=params.len()).map(|i| format_ident!("a{}", i)).collect();

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
