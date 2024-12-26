pub fn pack_kernel_args(
    arg_ptrs: *mut *mut std::ffi::c_void,
    info: &[crate::elf::KernelParamInfo],
) -> Box<[u8]> {
    let Some(last) = info.last() else { return Default::default() };
    let mut result = vec![0u8; (last.offset + last.size()) as usize];
    for (param, arg_ptr) in
        info.iter().zip(unsafe { std::slice::from_raw_parts(arg_ptrs, info.len()) })
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                arg_ptr.cast(),
                result.as_mut_ptr().wrapping_add(param.offset as usize),
                param.size() as usize,
            );
        }
    }
    result.into_boxed_slice()
}
