use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    Ident, Type, LitInt, LitStr, Result, Token,
};
extern crate lazy_static;
use lazy_static::lazy_static;

pub enum ElementType {
    Void,
    Type(syn::Type),
}

impl quote::ToTokens for ElementType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            ElementType::Void => {
                let void_ident = quote! { () };
                void_ident.to_tokens(tokens)
            },
            ElementType::Type(ty) => ty.to_tokens(tokens),
        }
    }
}

impl Clone for ElementType {
    fn clone(&self) -> Self {
        match self {
            ElementType::Void => ElementType::Void,
            ElementType::Type(ty) => ElementType::Type(ty.clone()),
        }
    }
}

impl ElementType {
    pub fn get_bytes(&self) -> usize {
        match self {
            ElementType::Void => 0,
            ElementType::Type(_) => std::mem::size_of::<Type>(),
        }
    }
}

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
    pub ty: ElementType,
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
        let result_ty = match result_ty.as_str() {
            "" => ElementType::Void,
            _ => ElementType::Type(syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result")),
        };
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
            let ty = ElementType::Type(syn::parse_str::<Type>(&ty_str).expect("Expected valid type"));
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
        let result_ty = match result_ty.as_str() {
            "" => ElementType::Void,
            _ => ElementType::Type(syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result")),
        };
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
            let ty = ElementType::Type(syn::parse_str::<Type>(&ty_str).expect("Expected valid type"));
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
        let result_ty = match result_ty.as_str() {
            "" => ElementType::Void,
            _ => ElementType::Type(syn::parse_str::<Type>(&result_ty).expect("Expected valid type for result")),
        };
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
            let ty = ElementType::Type(syn::parse_str::<Type>(&ty_str).expect("Expected valid type"));
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

lazy_static! {
    pub static ref SHADOW_DESC_TYPES: Vec<String> = {
        vec![
            "cudnnTensorDescriptor_t".to_string(),
            "cudnnFilterDescriptor_t".to_string(),
            "cudnnConvolutionDescriptor_t".to_string(),
        ]
    };
}

pub fn get_success_status(ty: &str) -> &str {
    match ty {
        "cublasStatus_t" => "CUBLAS_STATUS_SUCCESS",
        "CUresult" => "CUDA_SUCCESS",
        "cudaError_t" => "cudaSuccess",
        "cudnnStatus_t" => "CUDNN_STATUS_SUCCESS",
        "nvmlReturn_t" => "NVML_SUCCESS",
        &_ => todo!(),
    }
}