//! Syntax parsing of hook definitions.
//!
//! This is a separate crate because we can't export normal items in a proc-macro crate.

use proc_macro2::TokenStream;
use quote::{quote, TokenStreamExt as _};
use syn::meta::{parser, ParseNestedMeta};
use syn::parse::{Parse, ParseStream, Parser};
use syn::spanned::Spanned;
use syn::{
    Attribute, Block, Error, Expr, ExprBlock, FnArg, LitInt, Result, Signature, Stmt, Token, Type,
    Visibility,
};

pub struct HookAttrs {
    pub proc_id: LitInt,
    pub is_async_api: bool,
    pub min_cuda_version: u8,
    pub max_cuda_version: u8,
}

impl HookAttrs {
    const ERROR: &str = "expected #[cuda_hook(proc_id = ...)] or #[cuda_custom_hook]";

    pub fn from_macro(args: TokenStream) -> Result<Self> {
        let span = args.span();
        let mut raw = RawAttrs::default();
        Parser::parse2(parser(|meta| raw.parse(meta)), args)?;
        raw.validate()
            .ok_or_else(|| Error::new(span, Self::ERROR))
    }

    pub fn from_attr(attr: &Attribute) -> Result<Self> {
        let mut raw = RawAttrs::default();
        attr.parse_nested_meta(|meta| raw.parse(meta))?;
        raw.validate()
            .ok_or_else(|| Error::new_spanned(attr, Self::ERROR))
    }
}

struct RawAttrs {
    proc_id: Option<LitInt>,
    is_async_api: bool,
    min_cuda_version: u8,
    max_cuda_version: u8,
}

impl Default for RawAttrs {
    fn default() -> Self {
        Self {
            proc_id: None,
            is_async_api: false,
            min_cuda_version: u8::MIN,
            max_cuda_version: u8::MAX,
        }
    }
}

impl RawAttrs {
    fn parse(&mut self, meta: ParseNestedMeta<'_>) -> Result<()> {
        match meta.path.require_ident()?.to_string().as_str() {
            "proc_id" => self.proc_id = Some(meta.value()?.parse()?),
            "async_api" => self.is_async_api = true,
            "min_cuda_version" => {
                self.min_cuda_version = meta.value()?.parse::<LitInt>()?.base10_parse()?
            }
            "max_cuda_version" => {
                self.max_cuda_version = meta.value()?.parse::<LitInt>()?.base10_parse()?
            }
            _ => return Err(meta.error("unsupported property")),
        }
        Ok(())
    }

    fn validate(self) -> Option<HookAttrs> {
        self.proc_id.map(|proc_id| HookAttrs {
            proc_id,
            is_async_api: self.is_async_api,
            min_cuda_version: self.min_cuda_version,
            max_cuda_version: self.max_cuda_version,
        })
    }
}

pub struct HookFnItem {
    pub sig: Signature,
    pub injections: HookInjections,
}

#[derive(Default)]
pub struct HookInjections {
    pub client_before_send: Option<Block>,
    pub client_after_recv: Option<Block>,
}

impl Parse for HookFnItem {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        check_before_signature(input)?;
        let sig: Signature = input.parse()?;
        let mut injections = HookInjections::default();
        if input.peek(Token![;]) {
            let _: Token![;] = input.parse().unwrap();
            return Ok(Self { sig, injections });
        }
        let block: Block = input.parse()?;
        if block.stmts.is_empty() {
            return Err(Error::new_spanned(block, "replace empty block with `;`"));
        }
        for stmt in block.stmts {
            let Stmt::Expr(Expr::Block(ExprBlock { attrs, label: Some(label), block }), _) = stmt
            else {
                return Err(Error::new_spanned(stmt, "expected 'section: { ... }"));
            };
            check_max_attributes(&attrs, 0)?;
            if block.stmts.is_empty() {
                return Err(Error::new_spanned(block, "empty injection block"));
            }
            match label.name.ident.to_string().as_str() {
                "client_before_send" => injections.client_before_send = Some(block),
                "client_after_recv" => injections.client_after_recv = Some(block),
                _ => {
                    return Err(Error::new_spanned(
                        label.name.ident,
                        "unsupported injection section",
                    ))
                }
            }
        }
        Ok(Self { sig, injections })
    }
}

pub struct CustomHookFn(Signature);

impl Parse for CustomHookFn {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        check_before_signature(input)?;
        let sig: Signature = input.parse()?;
        let _: Token![;] = input.parse()?;
        for arg in sig.inputs.iter() {
            let FnArg::Typed(arg) = arg else { panic!() };
            if !arg.attrs.is_empty() {
                return Err(Error::new_spanned(
                    &arg.attrs[0],
                    "custom hook should not have param attributes",
                ));
            }
        }
        Ok(Self(sig))
    }
}

impl CustomHookFn {
    pub fn to_plain_fn(&self) -> TokenStream {
        let sig = &self.0;
        quote! { #sig { unimplemented!() } }
    }
}

pub fn is_hacked_type(mut ty: &Type) -> bool {
    while let Type::Ptr(ptr) = ty {
        ty = &ptr.elem;
    }
    if let Type::Path(ty) = ty {
        if let Some(seg) = ty.path.segments.last() {
            return seg.ident.to_string().starts_with("Hacked");
        }
    }
    false
}

pub fn check_max_attributes(attrs: &[Attribute], max: usize) -> Result<()> {
    if attrs.len() <= max {
        Ok(())
    } else {
        let mut tokens = TokenStream::new();
        tokens.append_all(&attrs[max..]);
        Err(Error::new_spanned(
            tokens,
            format!("too many attributes, expected {max}"),
        ))
    }
}

fn check_before_signature(input: ParseStream<'_>) -> Result<()> {
    check_max_attributes(&input.call(Attribute::parse_outer)?, 0)?;
    match input.parse()? {
        Visibility::Inherited => Ok(()),
        vis => Err(Error::new_spanned(vis, "no visibility allowed")),
    }
}
