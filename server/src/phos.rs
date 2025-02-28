#[link(name = "pos")]
extern "C" {
    pub fn pos_create_workspace_cuda() -> *mut std::ffi::c_void;
    fn pos_process(
        pos_cuda_ws: *mut std::ffi::c_void,
        api_id: u64,
        uuid: u64,
        param_desps: *mut u64,
        param_num: i32,
    ) -> i32;
    pub fn pos_destory_workspace_cuda(pos_cuda_ws: *mut std::ffi::c_void) -> i32;
}

#[cfg(target_pointer_width = "64")]
pub fn call_pos_process(
    pos_cuda_ws: *mut std::ffi::c_void,
    api_id: i32,
    uuid: u64,
    param_desps: &[usize],
) -> i32 {
    unsafe {
        pos_process(
            pos_cuda_ws,
            api_id as u64,
            uuid,
            param_desps.as_ptr() as *mut u64,
            (param_desps.len() / 2) as i32,
        )
    }
}
