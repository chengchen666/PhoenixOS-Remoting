#[link(name = "pos")]
extern "C" {
    pub fn pos_create_agent() -> *mut std::ffi::c_void;
    pub fn pos_destory_agent(pos_agent: *mut std::ffi::c_void) -> i32;
}
