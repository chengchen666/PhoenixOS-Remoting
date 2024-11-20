#![cfg_attr(not(feature = "phos"), expect(unused_variables))]

use super::*;
use cudasys::cudnn::*;

pub fn cudnnGetErrorStringExe<C: CommChannel>(
    proc_id: i32,
    server: &mut ServerWorker<C>,
) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    log::debug!("[{}:{}] cudnnGetErrorString", std::file!(), std::line!());
    let mut status: cudnnStatus_t = Default::default();
    if let Err(e) = status.recv(channel_receiver) {
        error!("Error receiving status: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    #[cfg(not(feature = "phos"))]
    let result = unsafe { cudnnGetErrorString(status) };
    #[cfg(feature = "phos")]
    let result = pos_process(
        POS_CUDA_WS.lock().unwrap().get_ptr(),
        proc_id,
        0u64,
        &[
            addr_of(&status), size_of_val(&status),
        ],
        0u64,
        0u64,
    ) as *const i8;
    // transfer to Vec<u8>
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
