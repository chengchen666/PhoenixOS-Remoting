use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use hookdef::{is_hacked_type, HookAttrs};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned as _;
use syn::{
    parse_quote, Attribute, FnArg, ForeignItem, Ident, Item, ItemFn, Meta, Signature, Token,
    Type, UseTree, Visibility,
};

pub fn generate_impls(
    hooks_path: &str,
    bindings_dir: &str,
    output_path: &str,
    unimplement_path: Option<&str>,
    cuda_version: u8,
) {
    let target_attr = match unimplement_path {
        Some(_) => "cuda_hook_hijack",
        None => "cuda_hook_exe",
    };
    let dir = fs::read_dir(bindings_dir).expect("failed to read bindings directory");
    for entry in dir {
        let path = entry.unwrap().path();
        let (module, extension) = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split_once('.')
            .unwrap();
        assert_eq!(extension, "rs");
        let mut bindings = parse_bindings(&path);
        let hooks_path = hooks_path.replace("{}", module);
        let comment = ["// Generated from ", &hooks_path, "\n\n"].concat();
        let imports = {
            let module = Ident::new(module, Span::call_site());
            match unimplement_path {
                Some(_) => [
                    Item::Use(parse_quote! { use super::*; }),
                    Item::Use(parse_quote! { use cudasys::types::#module::*; }),
                ],
                None => [
                    Item::Use(parse_quote! { use super::*; }),
                    Item::Use(parse_quote! { use cudasys::#module::*; }),
                ],
            }
        };
        convert_hooks(
            &hooks_path,
            &output_path.replace("{}", module),
            &comment,
            &imports,
            &mut bindings,
            target_attr,
            cuda_version,
        );
        if let Some(unimplement_path) = unimplement_path {
            gen_unimplement(
                &unimplement_path.replace("{}", module),
                &comment,
                &imports[1..],
                bindings,
            );
        }
    }
}

fn parse_bindings(path: &Path) -> BTreeMap<String, Signature> {
    eprintln!("parsing {path:?}");
    let bindings = fs::read_to_string(path).unwrap();
    let file = syn::parse_file(&bindings).unwrap();
    let mut result = BTreeMap::new();
    for item in file.items {
        let Item::ForeignMod(foreign) = item else {
            panic!()
        };
        for item in foreign.items {
            let ForeignItem::Fn(func) = item else {
                panic!()
            };
            let name = func.sig.ident.to_string();
            result.insert(name, func.sig);
        }
    }
    result
}

fn convert_hooks(
    hooks_path: &str,
    output_path: &str,
    comment: &str,
    imports: &[Item],
    bindings: &mut BTreeMap<String, Signature>,
    target_attr: &str,
    cuda_version: u8,
) {
    eprintln!("parsing {hooks_path:?}");
    let hooks = fs::read_to_string(hooks_path).unwrap();
    let file = syn::parse_file(&hooks).unwrap();
    let mut output = imports.to_vec();

    for item in file.items {
        match item {
            Item::Use(ref use_item) => {
                if let UseTree::Path(path) = &use_item.tree {
                    if path.ident == "std" {
                        output.push(item);
                    }
                }
            }
            Item::Type(_) => output.push(item),
            Item::Fn(mut func) => {
                let binding = bindings.remove(&func.sig.ident.to_string());
                if check_sig_replace_attr(
                    &mut func.attrs,
                    &func.sig,
                    binding,
                    target_attr,
                    cuda_version,
                    &mut output,
                ) {
                    output.push(Item::Fn(func));
                }
            }
            Item::Verbatim(tokens) => {
                let mut func: HookDef = syn::parse2(tokens).unwrap();
                let binding = bindings.remove(&func.sig.ident.to_string());
                if check_sig_replace_attr(
                    &mut func.attrs,
                    &func.sig,
                    binding,
                    target_attr,
                    cuda_version,
                    &mut output,
                ) {
                    output.push(Item::Verbatim(func.to_token_stream()));
                }
            }
            _ => {
                output.push(Item::Macro(parse_quote! {
                    compile_error!("unexpected item below");
                }));
                output.push(item);
            }
        }
    }

    let output = prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: vec![parse_quote! { #![allow(non_snake_case)] }],
        items: output,
    });

    let mut output_file = fs::File::create(output_path).unwrap();
    output_file.write_all(comment.as_bytes()).unwrap();
    output_file.write_all(output.as_bytes()).unwrap();
}

/// Returns true if the function should be emitted.
fn check_sig_replace_attr(
    attrs: &mut [Attribute],
    sig: &Signature,
    binding: Option<Signature>,
    target_attr: &str,
    cuda_version: u8,
    output: &mut Vec<Item>,
) -> bool {
    let err_item = || {
        Item::Macro(parse_quote! {
            compile_error!("unrecognized hook definition, expected 1 attribute");
        })
    };
    if attrs.len() != 1 {
        output.push(err_item());
        return true;
    }

    let attr = &mut attrs[0];
    match attr.path().segments.last().unwrap().ident.to_string().as_str() {
        "cuda_hook" => {}
        "cuda_custom_hook" => {
            eprintln!("skipped custom hook `{}`", sig.ident);
            return false;
        }
        _ => {
            output.push(err_item());
            return true;
        }
    }
    match HookAttrs::from_attr(attr) {
        Ok(attrs) => {
            if cuda_version < attrs.min_cuda_version || attrs.max_cuda_version < cuda_version {
                println!(
                    "cargo:warning=not emitting hook for `{}` because it's incompatible with CUDA {}",
                    sig.ident, cuda_version
                );
                return false;
            }
        }
        Err(err) => {
            output.push(Item::Macro(syn::parse2(err.to_compile_error()).unwrap()));
            return true;
        }
    }
    match attr.meta {
        Meta::List(ref mut meta) => {
            meta.path = Ident::new(target_attr, meta.path.span()).into();
        }
        _ => unreachable!(),
    }

    let Some(mut binding) = binding else {
        output.push(Item::Macro(parse_quote! {
            compile_error!("binding not found for the function below");
        }));
        return true;
    };

    if !is_sig_equal_ignore_attr(sig, &binding) {
        output.push(Item::Macro(parse_quote! {
            compile_error!("function signature mismatch");
        }));
        binding.ident = format_ident!("_binding__{}", binding.ident);
        output.push(Item::Fn(ItemFn {
            attrs: vec![parse_quote! { #[expect(unused_variables)] }],
            vis: Visibility::Inherited,
            sig: binding,
            block: parse_quote!({ unimplemented!() }),
        }));
        return true;
    };

    true
}

fn is_sig_equal_ignore_attr(hook: &Signature, binding: &Signature) -> bool {
    if hook.constness != binding.constness
        || hook.asyncness != binding.asyncness
        || hook.unsafety != binding.unsafety
        || hook.abi != binding.abi
        || hook.ident != binding.ident
        || hook.generics != binding.generics
        || hook.variadic != binding.variadic
        || hook.output != binding.output
    {
        return false;
    }

    if hook.inputs.len() != binding.inputs.len() {
        return false;
    }

    for pair in hook.inputs.iter().zip(binding.inputs.iter()) {
        let (FnArg::Typed(hook), FnArg::Typed(binding)) = pair else {
            return false;
        };
        if hook.pat != binding.pat {
            return false;
        }
        if !is_type_equal_ignore_path(&hook.ty, &binding.ty) && !is_hacked_type(&hook.ty) {
            return false;
        }
    }
    true
}

fn is_type_equal_ignore_path(hook: &Type, binding: &Type) -> bool {
    match (hook, binding) {
        (Type::Ptr(hook), Type::Ptr(binding)) => {
            hook.const_token == binding.const_token
                && hook.mutability == binding.mutability
                && is_type_equal_ignore_path(&hook.elem, &binding.elem)
        }
        (Type::Path(hook), Type::Path(binding)) => {
            hook.path.segments.last() == binding.path.segments.last()
        }
        _ => panic!(
            "unsupported type comparison: {} vs {}",
            hook.to_token_stream(),
            binding.to_token_stream()
        ),
    }
}

fn gen_unimplement(
    output_path: &str,
    comment: &str,
    imports: &[Item],
    bindings: BTreeMap<String, Signature>,
) {
    let mut items = imports.to_vec();
    let attrs = vec![parse_quote! { #[no_mangle] }];
    let abi = Some(parse_quote!(extern "C"));
    for mut sig in bindings.into_values() {
        let name = sig.ident.to_string();
        sig.abi = abi.clone();
        items.push(Item::Fn(ItemFn {
            attrs: attrs.clone(),
            vis: Visibility::Public(Default::default()),
            sig,
            block: parse_quote!({ unimplemented!(#name) }),
        }));
    }
    let output = prettyplease::unparse(&syn::File {
        shebang: None,
        attrs: vec![parse_quote! { #![allow(unused_variables)] }],
        items,
    });
    let mut file = fs::File::create(output_path).unwrap();
    file.write_all(comment.as_bytes()).unwrap();
    file.write_all(output.as_bytes()).unwrap();
}

struct HookDef {
    attrs: Vec<Attribute>,
    sig: Signature,
    semi: Token![;],
}

impl Parse for HookDef {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self {
            attrs: input.call(Attribute::parse_outer)?,
            sig: input.parse()?,
            semi: input.parse()?,
        })
    }
}

impl ToTokens for HookDef {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.attrs[0].to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
