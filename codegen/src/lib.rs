#![feature(proc_macro_diagnostic)]

use hookdef::{is_hacked_type, CustomHookAttrs, CustomHookFn};
use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_each_token_spanned, quote_spanned, TokenStreamExt as _};
use syn::{parse_macro_input, Type};

mod utils;
use utils::{define_usize_from, is_shadow_desc_type, ElementMode, PassBy};

mod hook_fn;
use hook_fn::HookFn;

mod use_thread_local;

/// Basic checks on a hook declaration
#[proc_macro_attribute]
pub fn cuda_hook(args: TokenStream, input: TokenStream) -> TokenStream {
    match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func.into_plain_fn().into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn cuda_custom_hook(args: TokenStream, input: TokenStream) -> TokenStream {
    if let Err(err) = CustomHookAttrs::from_macro(args.into()) {
        return err.to_compile_error().into();
    }
    parse_macro_input!(input as CustomHookFn)
        .to_plain_fn()
        .into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The derive macros for auto trait impl.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The derive procedural macro to generate `Transportable` trait implementation for a *fixed-size* type.
///
/// ### Example
/// To use this macro, annotate a struct or enum with `#[derive(Transportable)]`.
///
/// ```ignore
/// #[derive(Transportable)]
/// pub struct MyStruct {
///     ...
/// }
/// ```
///
/// This invocation generates a `Transportable` trait implementation for `MyStruct`.
/// The `Transportable` trait is used for sending and receiving the struct over a communication channel.
#[proc_macro_derive(Transportable)]
pub fn transportable_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    let gen = quote! {
        impl network::TransportableMarker for #name {}
    };

    gen.into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for client usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate hijack functions for client intercepting application calls.
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
/// #[cuda_hook_hijack(proc_id = 0)]
/// fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// This invocation generates a function `cudaGetDevice` with the same signature as the original function,
/// which will intercept the call to `cudaGetDevice` and send the parameters to the server then wait for the result.
#[proc_macro_attribute]
pub fn cuda_hook_hijack(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };

    let (proc_id, func, result, params) = (input.proc_id, input.func, input.result, input.params);

    let vars: Box<_> = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .collect();

    // send parameters
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            let deref = match &param.pass_by {
                PassBy::InputValue => Default::default(),
                PassBy::SinglePtr => {
                    if is_hacked_type(&param.ty) {
                        quote! { let #name = unsafe { std::ptr::read_unaligned(#name) }; }
                    } else {
                        quote! { let #name = unsafe { *#name }; }
                    }
                }
                PassBy::ArrayPtr { len, .. } => {
                    let len_ident = format_ident!("{}_len", name);
                    let mut tokens = define_usize_from(&len_ident, len);
                    let span = name.span();
                    quote_each_token_spanned! {tokens span
                        let #name = unsafe { std::slice::from_raw_parts(#name, #len_ident) };
                        match send_slice(#name, channel_sender) {
                            Ok(()) => {}
                            Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                        }
                    };
                    return tokens;
                }
                PassBy::InputCStr => {
                    return quote_spanned! {name.span()=>
                        let #name = unsafe { std::ffi::CStr::from_ptr(#name) };
                        match send_slice(#name.to_bytes_with_nul(), channel_sender) {
                            Ok(()) => {}
                            Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                        }
                    }
                }
            };
            quote_spanned! {name.span()=>
                #deref
                log::trace!("(input) {} = {:?}", stringify!(#name), #name);
                match #name.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                }
            }
        });

    // receive vars
    let recv_statements = vars.iter().map(|var| {
        let name = &var.name;
        let deref = match &var.pass_by {
            PassBy::InputValue | PassBy::InputCStr => unreachable!(),
            PassBy::SinglePtr => quote! { let #name = unsafe { &mut *#name }; },
            PassBy::ArrayPtr { len, .. } => {
                let len_ident = format_ident!("{}_len", name);
                let mut tokens = define_usize_from(&len_ident, len);
                let span = name.span();
                quote_each_token_spanned! {tokens span
                    let #name = unsafe { std::slice::from_raw_parts_mut(#name, #len_ident) };
                    match recv_slice_to(#name, channel_receiver) {
                        Ok(()) => {}
                        Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                    }
                };
                return tokens;
            }
        };
        quote_spanned! {name.span()=>
            // FIXME: allocate space for null pointers
            #deref
            match #name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
            }
            log::trace!("(output) {} = {:?}", stringify!(#name), #name);
        }
    });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = &param.ty;
        quote! { #name: #ty }
    });
    let result_name = &result.name;
    let result_ty = &result.ty;

    let async_api_return = if input.is_async_api {
        quote! {
            if cfg!(feature = "async_api") {
                return #result_name;
            }
        }
    } else {
        Default::default()
    };

    let (shadow_desc_send, shadow_desc_return) = if input.is_create_shadow_desc {
        let name = &vars[0].name;
        let shadow_desc_send = quote_spanned! {name.span()=>
            if cfg!(feature = "shadow_desc") {
                let resource_idx = client.resource_idx;
                unsafe {
                    *#name = resource_idx as _;
                }
                client.resource_idx += 1;
                match resource_idx.send(channel_sender) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                }
            }
        };
        let shadow_desc_return = quote! {
            if cfg!(feature = "shadow_desc") {
                return #result_name;
            }
        };
        (shadow_desc_send, shadow_desc_return)
    } else {
        Default::default()
    };

    let client_before_send = input.injections.client_before_send.iter();
    let client_extra_send = input.injections.client_extra_send.iter();
    let client_after_recv = input.injections.client_after_recv.iter();

    let gen_fn = quote! {
        #[no_mangle]
        // manually expanded the following macro to work around rust-analyzer bug
        // otherwise `Expand macro recursively at caret` is bugged
        // #[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
        CLIENT_THREAD.with_borrow_mut(|client| {
            log::debug!("[#{}] [{}:{}] {}", client.id, std::file!(), std::line!(), stringify!(#func));
            let ClientThread { channel_sender, channel_receiver, .. } = client;
            let proc_id: i32 = #proc_id;
            let mut #result_name: #result_ty = Default::default();

            #( #client_before_send )*

            match proc_id.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send {}: {}", "proc_id", e),
            }
            #( #send_statements )*
            #shadow_desc_send

            #( #client_extra_send )*

            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {}", e),
            }

            #shadow_desc_return
            #async_api_return

            #( #recv_statements )*
            match #result_name.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", stringify!(#result_name), e),
            }
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", "timestamp", e),
            }
            if #result_name != Default::default() {
                log::error!(
                    "{} returned error: {:?}\n{}",
                    stringify!(#func),
                    #result_name,
                    std::backtrace::Backtrace::force_capture(),
                );
            }
            #( #client_after_recv )*
            return #result_name;
        })
        }
    };

    gen_fn.into()
}

#[proc_macro_attribute]
pub fn use_thread_local(args: TokenStream, input: TokenStream) -> TokenStream {
    use_thread_local::use_thread_local(args.into(), input.into()).into()
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The collection of the procedural macro to generate Rust functions for server usage.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The procedural macro to generate execution functions for the server dispatcher.
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
/// #[cuda_hook_exe(proc_id = 1)]
/// fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
/// ```
///
/// This invocation generates a function `cudaSetDeviceExe` with `channel_sender` and `channel_receiver` for communication.
/// `cudaSetDeviceExe` will be called by the server dispatcher to execute the native `cudaSetDevice` function and send the result back to the client.
#[proc_macro_attribute]
pub fn cuda_hook_exe(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };

    let (func, result, params) = (input.func, input.result, input.params);
    let func_exe = format_ident!("{}Exe", func);

    // definition statements
    let def_statements = params.iter().map(|param| {
        if param.mode != ElementMode::Output {
            return Default::default();
        }
        let name = &param.name;
        let ptr_ident = param.get_exe_ptr_ident();
        let ty = &param.ty;
        let Type::Ptr(ptr) = ty else { panic!() };
        let ty = ptr.elem.as_ref();
        match &param.pass_by {
            PassBy::InputValue | PassBy::InputCStr => unreachable!(),
            PassBy::SinglePtr => quote_spanned! {name.span()=>
                let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
                let #ptr_ident = #name.as_mut_ptr();
            },
            PassBy::ArrayPtr { len, cap } => {
                let cap = cap.as_ref().unwrap_or(len);
                let cap_ident = format_ident!("{}_cap", name);
                let mut tokens = define_usize_from(&cap_ident, cap);
                let span = name.span();
                quote_each_token_spanned! {tokens span
                    let mut #name = Box::<[#ty]>::new_uninit_slice(#cap_ident);
                    let #ptr_ident = std::mem::MaybeUninit::slice_as_mut_ptr(&mut #name);
                };
                tokens
            }
        }
    });
    let assume_init = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .map(|param| {
            let name = &param.name;
            quote_spanned! {name.span()=>
                let #name = unsafe { #name.assume_init() };
                log::trace!("(output) {} = {:?}", stringify!(#name), #name);
            }
        });

    // receive parameters
    let recv_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            let recv_single = |ty| {
                quote_spanned! {name.span()=>
                    let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
                    match #name.recv(channel_receiver) {
                        Ok(()) => {}
                        Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
                    }
                    let #name = unsafe { #name.assume_init() };
                    log::trace!("(input) {} = {:?}", stringify!(#name), #name);
                }
            };
            match &param.pass_by {
                PassBy::InputValue => recv_single(&param.ty),
                PassBy::SinglePtr => {
                    let Type::Ptr(ptr) = &param.ty else { panic!() };
                    let mut tokens = recv_single(ptr.elem.as_ref());
                    let span = name.span();
                    let ptr_ident = param.get_exe_ptr_ident();
                    quote_each_token_spanned! {tokens span
                        let #ptr_ident = &raw const #name;
                    }
                    tokens
                }
                PassBy::ArrayPtr { .. } => {
                    let Type::Ptr(ptr) = &param.ty else { panic!() };
                    let ty = ptr.elem.as_ref();
                    let ptr_ident = param.get_exe_ptr_ident();
                    return quote_spanned! {name.span()=>
                        let #name = match recv_slice::<#ty, _>(channel_receiver) {
                            Ok(slice) => slice,
                            Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
                        };
                        log::trace!("(input) {} = {:p}", stringify!(#name), #name.as_ptr());
                        let #ptr_ident = #name.as_ptr();
                    };
                }
                PassBy::InputCStr => {
                    let ptr_ident = param.get_exe_ptr_ident();
                    quote_spanned! {name.span()=>
                        let #name = match recv_slice(channel_receiver) {
                            Ok(slice) => {
                                std::ffi::CString::from_vec_with_nul(slice.into_vec()).unwrap()
                            }
                            Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
                        };
                        let #ptr_ident = #name.as_ptr();
                    }
                }
            }
        });

    let is_destroy = func.to_string().contains("Destroy");

    // get resource when SR
    let get_resource_statements = params
        .iter()
        .filter(|param| {
            let ty = &param.ty;
            param.mode == ElementMode::Input && is_shadow_desc_type(ty)
        })
        .map(|param| {
            let name = &param.name;
            let ty = &param.ty;
            if is_destroy {
                assert_eq!(params.len(), 1);
                quote! {
                    #[cfg(feature = "shadow_desc")]
                    let #name = server.resources.remove(&(#name as usize)).unwrap() as #ty;
                }
            } else {
                quote! {
                    #[cfg(feature = "shadow_desc")]
                    let #name = *server.resources.get(&(#name as usize)).unwrap() as #ty;
                }
            }
        });

    // execution statement
    let result_name = &result.name;
    let exec_statement = if !input.injections.server_execution.is_empty() {
        let mut tokens = proc_macro2::TokenStream::new();
        tokens.append_all(&input.injections.server_execution);
        tokens
    } else {
        let result_ty = &result.ty;
        let exec_args = params.iter().map(|param| {
            let name = &param.name;
            let arg = if let PassBy::InputValue = param.pass_by {
                &name
            } else {
                &param.get_exe_ptr_ident()
            };
            if param.is_void_ptr || is_hacked_type(&param.ty) {
                quote_spanned!(name.span()=> (#arg).cast())
            } else {
                quote!(#arg)
            }
        });
        let pos_args = params
            .iter()
            .filter(|param| param.mode == ElementMode::Input)
            .map(|param| {
                let name = &param.name;
                let arg = if let PassBy::InputValue = param.pass_by {
                    &name
                } else {
                    &param.get_exe_ptr_ident()
                };
                match param.pass_by {
                    PassBy::ArrayPtr { .. } => quote_spanned![arg.span()=>
                        #arg as usize,
                        size_of_val(&#name.as_ref()),
                    ],
                    _ => quote_spanned![arg.span()=>
                        &raw const #name as usize,
                        size_of_val(&#name),
                    ],
                }
            });
        let get_pos_ret_data_len = params
            .iter()
            .filter(|param| param.mode == ElementMode::Output)
            .map(|param| {
                let name = &param.name;
                match param.pass_by {
                    PassBy::SinglePtr => quote! {
                        pos_ret_data_len += size_of_val(&#name);
                    },
                    PassBy::ArrayPtr { .. } => quote! {
                        pos_ret_data_len += size_of_val(&#name.as_ref());
                    },
                    _ => unreachable!()
                }
            });
        let copy_pos_output_params = params
            .iter()
            .filter(|param| param.mode == ElementMode::Output)
            .map(|param| {
                let name = &param.name;
                let arg = if let PassBy::InputValue = param.pass_by {
                    &name
                } else {
                    &param.get_exe_ptr_ident()
                };
                match param.pass_by {
                    PassBy::SinglePtr => quote! {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                pos_ret_data[pos_copy_offset..].as_ptr(),
                                #arg as *mut u8,
                                size_of_val(&#name)
                            );
                        }
                        pos_copy_offset += size_of_val(&#name);
                    },
                    PassBy::ArrayPtr { .. } => quote! {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                pos_ret_data[pos_copy_offset..].as_ptr(),
                                #arg as *mut u8,
                                size_of_val(&#name.as_ref())
                            );
                        }
                        pos_copy_offset += size_of_val(&#name.as_ref());
                    },
                    _ => unreachable!()
                }
            });
        #[cfg(not(feature = "phos"))]
        quote! {
            let #result_name: #result_ty = unsafe { #func(#(#exec_args),*) };
        }
        #[cfg(feature = "phos")]
        quote! {
            let mut pos_ret_data_len: usize = 0;
            #( #get_pos_ret_data_len )*
            let mut pos_ret_data = vec![0u8; pos_ret_data_len];
            let pos_ret_data_ptr: u64 = match pos_ret_data_len {
                0 => 0u64,
                _ => pos_ret_data.as_mut_ptr() as u64,
            };
            let #result_name = #result_ty::from_i32(pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                proc_id,
                0u64,
                &[#(#pos_args)*],
                pos_ret_data_ptr,
                pos_ret_data_len as u64,
            )).expect("Illegal result ID");
            let mut pos_copy_offset: usize = 0;
            #( #copy_pos_output_params )*
        }
    };

    // send result
    let send_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Output)
        .map(|param| {
            let name = &param.name;
            let send = match &param.pass_by {
                PassBy::InputValue | PassBy::InputCStr => unreachable!(),
                PassBy::SinglePtr => quote! { #name.send(channel_sender) },
                PassBy::ArrayPtr { len, .. } => {
                    let len_ident = format_ident!("{}_len", name);
                    let mut tokens = define_usize_from(&len_ident, len);
                    let span = name.span();
                    quote_each_token_spanned! {tokens span
                        send_slice(&#name[..#len_ident], channel_sender)
                    };
                    tokens
                }
            };
            quote_spanned! {name.span()=>
                match { #send } {
                    Ok(()) => {}
                    Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                }
            }
        });

    let (shadow_desc_recv, shadow_desc_return) = if input.is_create_shadow_desc {
        let name = &params[0].name;
        let shadow_desc_recv = quote_spanned! {name.span()=>
            #[cfg(feature = "shadow_desc")]
            let mut resource_idx = 0usize;
            #[cfg(feature = "shadow_desc")]
            match resource_idx.recv(channel_receiver) {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
            }
        };
        let shadow_desc_return = quote_spanned! {name.span()=>
            #[cfg(feature = "shadow_desc")]
            server.resources.insert(resource_idx, #name as usize);
            if cfg!(feature = "shadow_desc") {
                return;
            }
        };
        (shadow_desc_recv, shadow_desc_return)
    } else {
        Default::default()
    };

    let async_api_return = if input.is_async_api {
        quote! {
            if cfg!(feature = "async_api") {
                return;
            }
        }
    } else {
        Default::default()
    };

    let server_extra_recv = input.injections.server_extra_recv.iter();
    let server_after_send = input.injections.server_after_send.iter();

    let gen_fn = quote! {
        pub fn #func_exe<C: CommChannel>(proc_id: i32, server: &mut ServerWorker<C>) {
            let ServerWorker { channel_sender, channel_receiver, .. } = server;
            log::debug!("[#{}] [{}:{}] {}", server.id, std::file!(), std::line!(), stringify!(#func));
            #( #recv_statements )*
            #( #get_resource_statements )*
            #shadow_desc_recv
            #( #server_extra_recv )*
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive {}: {:?}", "timestamp", e),
            }
            #( #def_statements )*
            #exec_statement
            #( #assume_init )*

            if #result_name != Default::default() {
                log::error!("{} returned error: {:?}", stringify!(#func), #result_name);
            }

            #shadow_desc_return
            #async_api_return
            #( #send_statements )*
            match #result_name.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send {}: {}", stringify!(#result_name), e),
            }
            match channel_sender.flush_out() {
                Ok(()) => {}
                Err(e) => panic!("failed to flush_out: {}", e),
            }
            #( #server_after_send )*
        }
    };

    gen_fn.into()
}
