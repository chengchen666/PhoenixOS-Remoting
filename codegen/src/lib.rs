extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Ident};

mod utils;
use utils::{Element, ElementMode, ExeParser, HijackParser};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for client usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate hijack functions for client intercepting application calls.
///
/// To use this macro, annotate a call to `gen_hijack` with specified proc_id and the desired function name,
/// followed by the types of the return and parameters as string literals.
///
/// ### Example
/// We have a function `cudaGetDevice` with the following signature:
///
/// ```ignore
/// pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// To use this macro generating a hijack function for interception, we can write:
///
/// ```ignore
/// gen_hijack!("0", "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
/// ```
///
/// This invocation generates a function `cudaGetDevice` with the same signature as the original function,
/// which will intercept the call to `cudaGetDevice` and send the parameters to the server then wait for the result.
///
/// Specifically, the function is expanded as:
///
/// ```ignore
/// #[no_mangle]
/// pub extern "C" fn cudaGetDevice(param1: *mut ::std::os::raw::c_int) -> cudaError_t {
///     println!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
///     let proc_id = 0;
///     let mut var1 = Default::default();
///     let mut result = Default::default();
///
///     match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
///         Ok(_) => {}
///         Err(e) => panic!("failed to serialize proc_id: {:?}", e),
///     }
///     match CHANNEL_SENDER.lock().unwrap().flush_out() {
///         Ok(_) => {}
///         Err(e) => panic!("failed to send: {:?}", e),
///     }
///
///     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut var1) {
///         Ok(_) => {}
///         Err(e) => panic!("failed to deserialize var1: {:?}", e),
///     }
///     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
///         Ok(_) => {}
///         Err(e) => panic!("failed to deserialize result: {:?}", e),
///     }
///     unsafe {
///         *param1 = var1;
///     }
///     return result;
/// }
/// ```
///
#[proc_macro]
pub fn gen_hijack(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as HijackParser);

    let (proc_id, func, result, params) = (input.proc_id, input.func, input.result, input.params);

    let vars: Vec<Element> = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .enumerate()
        .map(|(i, param)| Element {
            name: format_ident!("var{}", i + 1),
            ty: param.ty.clone(),
            mode: ElementMode::Output,
        })
        .collect();

    // definition statements
    let def_statements = vars.iter().map(|var| {
        let name = &var.name;
        let ty = &var.ty;
        quote! { let mut #name: #ty = Default::default(); }
    });

    // send parameters
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            quote! {
                match CHANNEL_SENDER.lock().unwrap().send_var(&#name) {
                    Ok(_) => {}
                    Err(e) => panic!("failed to serialize #name: {:?}", e),
                }
            }
        });

    // receive vars
    let recv_statements = vars.iter().map(|var| {
        let name = &var.name;
        quote! {
            match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut #name) {
                Ok(_) => {}
                Err(e) => panic!("failed to deserialize #name: {:?}", e),
            }
        }
    });

    // assign vars to params
    let assign_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .zip(vars.iter())
        .map(|(param, var)| {
            let param_name = &param.name;
            let var_name = &var.name;
            quote! { unsafe { *#param_name = #var_name; } }
        });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        match param.mode {
            ElementMode::Input => quote! { #name: #ty },
            ElementMode::Output => quote! { #name: *mut #ty },
        }
    });
    let result_name = &result.name;
    let result_ty = &result.ty;
    let gen_fn = quote! {
        #[no_mangle]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
            println!("[{}:{}] {}", std::file!(), std::line!(), stringify!(#func));
            let proc_id = #proc_id;
            #( #def_statements )*
            let mut #result_name: #result_ty = Default::default();

            match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
                Ok(_) => {}
                Err(e) => panic!("failed to serialize proc_id: {:?}", e),
            }
            #( #send_statements )*
            match CHANNEL_SENDER.lock().unwrap().flush_out() {
                Ok(_) => {}
                Err(e) => panic!("failed to send: {:?}", e),
            }

            #( #recv_statements )*
            #( #assign_statements )*
            match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
                Ok(_) => {}
                Err(e) => panic!("failed to deserialize result: {:?}", e),
            }
            return result;
        }
    };

    gen_fn.into()
}

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
    let input = parse_macro_input!(input as ExeParser);

    let (func, result, params) = (input.func, input.result, input.params);
    let func_exe = Ident::new(&format!("{}Exe", func), func.span());

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
        syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
            Ident::new(&lit_str.value(), lit_str.span())
        }
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, arg)| {
            let ty = match arg {
                syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
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
        syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
            Ident::new(&lit_str.value(), lit_str.span())
        }
        _ => panic!("Expected first argument to be a string literal for function name"),
    };

    let params: Vec<_> = input
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, arg)| {
            let ty = match arg {
                syn::NestedMeta::Lit(syn::Lit::Str(lit_str)) => {
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
