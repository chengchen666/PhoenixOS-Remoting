use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    Ident, Type, LitInt, LitStr, Result, Token,
};

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

pub struct UnimplementParser {
    pub func: Ident,
    pub result: Element,
    pub params: Vec<Element>,
}

impl Parse for UnimplementParser {
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

        Ok(UnimplementParser {
            func,
            result,
            params,
        })
    }
}