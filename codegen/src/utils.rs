use quote::ToTokens;
use syn::{Expr, Ident, Type};

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
    pub ty: Type,
    pub mode: ElementMode,
    pub pass_by: PassBy,
}

pub enum PassBy {
    InputValue,
    SinglePtr,
    ArrayPtr { len: Expr, cap: Option<Expr> },
}

pub fn is_shadow_desc_type(ty: &Type) -> bool {
    [
        "cudnnTensorDescriptor_t",
        "cudnnFilterDescriptor_t",
        "cudnnConvolutionDescriptor_t",
    ]
    .contains(&ty.to_token_stream().to_string().as_str())
}

pub fn is_async_return_type(ty: &Type) -> bool {
    [
        "cublasStatus_t",
        "CUresult",
        "cudaError_t",
        "cudnnStatus_t",
        "nvmlReturn_t",
    ].contains(&ty.to_token_stream().to_string().as_str())
}

pub fn is_void_ptr(ptr: &syn::TypePtr) -> bool {
    if let Type::Path(elem) = ptr.elem.as_ref() {
        if let Some(seg) = elem.path.segments.last() {
            return seg.ident == "c_void";
        }
    }
    false
}
