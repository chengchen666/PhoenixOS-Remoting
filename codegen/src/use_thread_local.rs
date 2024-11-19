use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse2, Error, Expr, Pat, Result, Token};

pub fn use_thread_local(args: TokenStream, input: TokenStream) -> TokenStream {
    fn inner(args: TokenStream, input: TokenStream) -> Result<TokenStream> {
        let Var { pat, tls } = parse2(args)?;
        let mut tts: Vec<_> = input.into_iter().collect();
        let block = tts.pop().unwrap();
        Ok(quote! {
            #(#tts)* {
                #tls(|#pat| #block)
            }
        })
    }

    inner(args, input).unwrap_or_else(Error::into_compile_error)
}

struct Var {
    pat: Pat,
    tls: Expr,
}

impl Parse for Var {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let var = input.call(Pat::parse_single)?;
        let _: Token![=] = input.parse()?;
        let tls = input.parse()?;
        Ok(Self { pat: var, tls })
    }
}
