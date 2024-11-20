#[link(name = "pos")]
extern "C" {
    pub fn pos_create_workspace_cuda() -> *mut std::ffi::c_void;
    pub fn call_pos_process(
        pos_cuda_ws: *mut std::ffi::c_void,
        api_id: u64,
        uuid: u64,
        param_desps: *mut std::ffi::c_void,
        param_num: i32,
        ret_data: *mut std::ffi::c_void,
        ret_data_len: u64
    ) -> i32;
    pub fn pos_destory_workspace_cuda(pos_cuda_ws: *mut std::ffi::c_void) -> i32;
}

pub struct POSWorkspace(pub *mut std::ffi::c_void);
unsafe impl Send for POSWorkspace {}
unsafe impl Sync for POSWorkspace {}
impl POSWorkspace {
    pub fn get_ptr(&self) -> *mut std::ffi::c_void {
        self.0
    }
}

pub fn pos_process(
    pos_cuda_ws: *mut std::ffi::c_void,
    api_id: i32,
    uuid: u64,
    param_desps: &[usize],
    ret_data_ptr: u64,
    ret_data_len: u64
) -> i32 {
    unsafe {
        call_pos_process(
            pos_cuda_ws,
            api_id as u64,
            uuid,
            param_desps.as_ptr() as *mut std::ffi::c_void,
            (param_desps.len() / 2) as i32,
            ret_data_ptr as *mut std::ffi::c_void,
            ret_data_len
        )
    }
}
