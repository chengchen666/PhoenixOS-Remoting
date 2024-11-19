#![expect(non_snake_case)]

use super::*;
use cudasys::cudnn::*;

pub fn cudnnGetErrorStringExe<C: CommChannel>(
    server: &mut ServerWorker<C>,
) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    info!("[{}:{}] cudnnGetErrorString", std::file!(), std::line!());
    let mut status: cudnnStatus_t = Default::default();
    if let Err(e) = status.recv(channel_receiver) {
        error!("Error receiving status: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = unsafe { cudnnGetErrorString(status) };
    // transfer to Vec<u8>
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
