use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    Ident, Type, LitInt, LitStr, Result, Token,
    Signature, ReturnType,
    parse_str,
};
use quote::ToTokens;

/// - "type", - "*mut type"
/// the former is input to native function,
/// the latter is output from native function
#[derive(PartialEq, Eq)]
pub enum ElementMode {
    Input,
    Output,
}

pub struct Element {
    pub name: Ident,
    pub ty: syn::Type,
    pub mode: ElementMode,
}

pub struct ExeParser {
    pub func: Ident,
    pub result: Element,
    pub params: Vec<Element>,
}

impl Parse for ExeParser {
    fn parse(input: ParseStream) -> Result<Self> {
        let func = input.parse::<LitStr>().expect("Expected second argument to be a string literal for function name").value();
        let func = format_ident!("{}", func);

        let _comma: Option<Token![,]> = input.parse().ok();
        let result_ty = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
        let result_ty = syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result");
        let result = Element {
            name: format_ident!("result"),
            ty: result_ty,
            mode: ElementMode::Output,
        };

        let mut params = Vec::new();
        let mut i: usize = 0;
        while !input.is_empty() {
            let _comma: Option<Token![,]> = input.parse().ok();
            let mut ty_str = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
            let mode = if ty_str.starts_with("*mut ") {
                ty_str = ty_str.replace("*mut ", "");
                ElementMode::Output
            } else {
                ElementMode::Input
            };
            let ty = syn::parse_str::<Type>(&ty_str).expect("Expected valid type");
            params.push(Element {
                name: format_ident!("param{}", i + 1),
                ty,
                mode,
            });
            i += 1;
        }

        Ok(ExeParser {
            func,
            result,
            params,
        })
    }
}

pub struct HijackParser {
    pub proc_id: LitInt,
    pub func: Ident,
    pub result: Element,
    pub params: Vec<Element>,
}

impl Parse for HijackParser {
    fn parse(input: ParseStream) -> Result<Self> {
        let proc_id = input.parse::<LitInt>().expect("Expected valid proc_id");

        let _comma: Option<Token![,]> = input.parse().ok();
        let func = input.parse::<LitStr>().expect("Expected second argument to be a string literal for function name").value();
        let func = format_ident!("{}", func);

        let _comma: Option<Token![,]> = input.parse().ok();
        let result_ty = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
        let result_ty = syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result");
        let result = Element {
            name: format_ident!("result"),
            ty: result_ty,
            mode: ElementMode::Output,
        };

        let mut params = Vec::new();
        let mut i: usize = 0;
        while !input.is_empty() {
            let _comma: Option<Token![,]> = input.parse().ok();
            let mut ty_str = input.parse::<LitStr>().expect("Expected valid type as a string literal").value();
            let mode = if ty_str.starts_with("*mut ") {
                ty_str = ty_str.replace("*mut ", "");
                ElementMode::Output
            } else {
                ElementMode::Input
            };
            let ty = syn::parse_str::<Type>(&ty_str).expect("Expected valid type");
            params.push(Element {
                name: format_ident!("param{}", i + 1),
                ty,
                mode,
            });
            i += 1;
        }

        Ok(HijackParser {
            proc_id,
            func,
            result,
            params,
        })
    }
}

pub fn sig_parse(input: &str, proc_id: i32) -> Result<HijackParser> {
    let proc_id = syn::parse_str::<LitInt>(&proc_id.to_string()).expect("Expected valid proc_id");

    let s = input
        .trim_start_matches('"')
        .trim_end_matches('"')
        .trim_start_matches("pub ")
        .trim_end_matches(";");
    let sig = parse_str::<Signature>(s).expect("Failed to parse function signature");

    // get func name
    let func = sig.ident;

    // get func return type
    let result = Element {
        name: format_ident!("result"),
        ty: match sig.output {
            ReturnType::Type(_, t) => *t,
            ReturnType::Default => panic!("Unimplement no return func"),
        },
        mode: ElementMode::Output,
    };

    // get params
    let mut params = Vec::new();
    let mut i: usize = 0;
    for param in sig.inputs.iter() {
        if let syn::FnArg::Typed(pat_type) = param {
            let ty: &Type = &(*pat_type.ty);
            let mut ty_str = type_to_string(ty);
            let mode = if ty_str.starts_with("*mut ") {
                ty_str = ty_str.replace("*mut ", "");
                ElementMode::Output
            } else {
                ElementMode::Input
            };
            let ty = syn::parse_str::<Type>(&ty_str).expect("Expected valid type");
            params.push(Element {
                name: format_ident!("param{}", i + 1),
                ty,
                mode,
            });
            i += 1;
        }
    }

    Ok(HijackParser {
        proc_id,
        func,
        result,
        params,
    })
}

fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::Ptr(type_ptr) => {
            let const_token = if type_ptr.const_token.is_some() { "const " } else { "" };

            let mutability = if type_ptr.mutability.is_some() {
                "mut "
            } else {
                ""
            };
            let inner_type = type_to_string(&*type_ptr.elem);
            format!("*{}{}{}", const_token, mutability, inner_type)
        }
        Type::Path(_) => {
            ty.to_token_stream().to_string().replace(" ", "")
        }
        _ => {
            panic!("Unimplemented type {:#?}", ty);
        }
    }
}