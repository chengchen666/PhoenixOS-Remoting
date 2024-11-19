#![feature(proc_macro_diagnostic)]

use hookdef::{is_hacked_type, CustomHookFn, HookInjections};
use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_each_token_spanned, quote_spanned};
use syn::parse::Nothing;
use syn::spanned::Spanned as _;
use syn::{parse_macro_input, Type};

mod utils;
use utils::{is_shadow_desc_type, is_void_ptr, ElementMode, PassBy};

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
    parse_macro_input!(args as Nothing);
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
                    let Type::Ptr(ty) = &param.ty else { panic!() };
                    let ptr = if is_void_ptr(ty) {
                        quote!(#name as *const u8)
                    } else {
                        quote!(#name)
                    };
                    let len_ident = format_ident!("{}_len", name);
                    let mut tokens = quote_spanned! {len.span()=>
                        let #len_ident = usize::try_from((#len).clone()).unwrap();
                    };
                    let span = name.span();
                    quote_each_token_spanned! {tokens span
                        let #name = unsafe { std::slice::from_raw_parts(#ptr, #len_ident) };
                        match send_slice(#name, channel_sender) {
                            Ok(()) => {}
                            Err(e) => panic!("failed to send {}: {}", stringify!(#name), e),
                        }
                    };
                    return tokens;
                }
            };
            quote_spanned! {name.span()=>
                #deref
                log::debug!("(input) {} = {:?}", stringify!(#name), #name);
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
            PassBy::InputValue => unreachable!(),
            PassBy::SinglePtr => quote! { let #name = unsafe { &mut *#name }; },
            PassBy::ArrayPtr { len, .. } => {
                let Type::Ptr(ty) = &var.ty else { panic!() };
                let ptr = if is_void_ptr(ty) {
                    quote!(#name as *mut u8)
                } else {
                    quote!(#name)
                };
                let len_ident = format_ident!("{}_len", name);
                let mut tokens = quote_spanned! {len.span()=>
                    let #len_ident = usize::try_from((#len).clone()).unwrap();
                };
                let span = name.span();
                quote_each_token_spanned! {tokens span
                    let #name = unsafe { std::slice::from_raw_parts_mut(#ptr, #len_ident) };
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
            log::debug!("(output) {} = {:?}", stringify!(#name), #name);
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
                    *#name = resource_idx;
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

    let HookInjections { client_before_send, client_after_recv } = input.injections;

    let gen_fn = quote! {
        #[no_mangle]
        #[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
        pub extern "C" fn #func(#(#params),*) -> #result_ty {
            info!("[#{}] [{}:{}] {}", client.id, std::file!(), std::line!(), stringify!(#func));
            let ClientThread { channel_sender, channel_receiver, .. } = client;
            let proc_id: i32 = #proc_id;
            let mut #result_name: #result_ty = Default::default();

            #client_before_send

            match proc_id.send(channel_sender) {
                Ok(()) => {}
                Err(e) => panic!("failed to send {}: {}", "proc_id", e),
            }
            #( #send_statements )*
            #shadow_desc_send

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
                log::warn!("{} returned error: {:?}", stringify!(#func), #result_name);
            }
            #client_after_recv
            return #result_name;
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
        if param.mode == ElementMode::Input {
            return Default::default();
        }
        let name = &param.name;
        let ty = &param.ty;
        let Type::Ptr(ptr) = ty else { panic!() };
        let ty = ptr.elem.as_ref();
        match &param.pass_by {
            PassBy::InputValue => unreachable!(),
            PassBy::SinglePtr => quote_spanned! {name.span()=>
                let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
            },
            PassBy::ArrayPtr { len, cap } => {
                let cap = cap.as_ref().unwrap_or(len);
                let cap_ident = format_ident!("{}_cap", name);
                let mut tokens = quote_spanned! {cap.span()=>
                    let #cap_ident = usize::try_from((#cap).clone()).unwrap();
                };
                let span = name.span();
                quote_each_token_spanned! {tokens span
                    let mut #name = Box::<[#ty]>::new_uninit_slice(#cap_ident);
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
                log::debug!("(output) {} = {:?}", stringify!(#name), #name);
            }
        });

    // receive parameters
    let recv_statements = params
        .iter()
        .filter(|param| param.mode == ElementMode::Input)
        .map(|param| {
            let name = &param.name;
            let ty = match &param.pass_by {
                PassBy::InputValue => &param.ty,
                PassBy::SinglePtr => {
                    let Type::Ptr(ptr) = &param.ty else { panic!() };
                    ptr.elem.as_ref()
                }
                PassBy::ArrayPtr { .. } => {
                    let Type::Ptr(ptr) = &param.ty else { panic!() };
                    let ty = ptr.elem.as_ref();
                    return quote_spanned! {name.span()=>
                        let #name = match recv_slice::<#ty, _>(channel_receiver) {
                            Ok(slice) => slice,
                            Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
                        };
                    };
                }
            };
            quote_spanned! {name.span()=>
                let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
                match #name.recv(channel_receiver) {
                    Ok(()) => {}
                    Err(e) => panic!("failed to receive {}: {}", stringify!(#name), e),
                }
                let #name = unsafe { #name.assume_init() };
                log::debug!("(input) {} = {:?}", stringify!(#name), #name);
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
            if is_destroy {
                assert_eq!(params.len(), 1);
                quote! {
                    #[cfg(feature = "shadow_desc")]
                    let #name = server.resources.remove(&(#name as usize)).unwrap();
                }
            } else {
                quote! {
                    #[cfg(feature = "shadow_desc")]
                    let #name = *server.resources.get(&(#name as usize)).unwrap();
                }
            }
        });

    // execution statement
    let result_name = &result.name;
    let exec_statement = {
        let result_ty = &result.ty;
        let params = params.iter().map(|param| {
            let name = &param.name;
            let arg = match param.mode {
                ElementMode::Input => match &param.pass_by {
                    PassBy::InputValue => quote!(#name),
                    PassBy::SinglePtr => quote_spanned!(name.span()=> &raw const #name),
                    PassBy::ArrayPtr { .. } => quote_spanned!(name.span()=> #name.as_ptr()),
                },
                ElementMode::Output => match &param.pass_by {
                    PassBy::InputValue => unreachable!(),
                    PassBy::SinglePtr => quote_spanned!(name.span()=> #name.as_mut_ptr()),
                    PassBy::ArrayPtr { .. } => quote_spanned! {name.span()=>
                        std::mem::MaybeUninit::slice_as_mut_ptr(&mut #name)
                    },
                },
            };
            if is_hacked_type(&param.ty) {
                quote_spanned!(name.span()=> std::mem::transmute(#arg))
            } else {
                arg
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
            let send = match &param.pass_by {
                PassBy::InputValue => unreachable!(),
                PassBy::SinglePtr => quote! { #name.send(channel_sender) },
                PassBy::ArrayPtr { len, .. } => {
                    let len_ident = format_ident!("{}_len", name);
                    let mut tokens = quote_spanned! {len.span()=>
                        let #len_ident = usize::try_from((#len).clone()).unwrap();
                    };
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

    let gen_fn = quote! {
        #[allow(non_snake_case)]
        pub fn #func_exe<C: CommChannel>(server: &mut ServerWorker<C>) {
            let ServerWorker { channel_sender, channel_receiver, .. } = server;
            info!("[#{}] [{}:{}] {}", server.id, std::file!(), std::line!(), stringify!(#func));
            #( #recv_statements )*
            #( #get_resource_statements )*
            #shadow_desc_recv
            match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e)
            }
            #( #def_statements )*
            #exec_statement
            #( #assume_init )*

            if #result_name != Default::default() {
                log::warn!("{} returned error: {:?}", stringify!(#func), #result_name);
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
        }
    };

    gen_fn.into()
}
