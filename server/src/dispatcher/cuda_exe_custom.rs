#![allow(non_snake_case)]

use super::*;
use cudasys::cuda::*;

pub fn __cudaRegisterFatBinaryExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] __cudaRegisterFatBinary",
        std::file!(),
        std::line!()
    );
    let mut fatbin: Vec<u8> = Default::default();
    fatbin.recv(channel_receiver).unwrap();
    let mut client_address: MemPtr = Default::default();
    client_address.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let mut module: CUmodule = Default::default();
    let result =
        unsafe { cuModuleLoadData(&mut module, fatbin.as_ptr() as *const std::os::raw::c_void) };
    add_module(client_address, module);

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

// TODO: We should also remove associated function handles
pub fn __cudaUnregisterFatBinaryExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] __cudaUnregisterFatBinary",
        std::file!(),
        std::line!()
    );
    let mut client_address: MemPtr = Default::default();
    client_address.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let module = get_module(client_address).unwrap();
    let result = unsafe { cuModuleUnload(module) };

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn __cudaRegisterFunctionExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] __cudaRegisterFunction", std::file!(), std::line!());
    let mut fatCubinHandle: MemPtr = Default::default();
    fatCubinHandle.recv(channel_receiver).unwrap();
    let mut hostFun: MemPtr = Default::default();
    hostFun.recv(channel_receiver).unwrap();
    let mut deviceName: Vec<u8> = Default::default();
    deviceName.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let mut device_func: CUfunction = Default::default();

    let module = get_module(fatCubinHandle).unwrap();
    let result = unsafe {
        cuModuleGetFunction(
            &mut device_func,
            module,
            deviceName.as_ptr() as *const std::os::raw::c_char,
        )
    };
    add_function(hostFun, device_func);

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn __cudaRegisterVarExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] __cudaRegisterVar", std::file!(), std::line!());
    let mut fatCubinHandle: MemPtr = Default::default();
    fatCubinHandle.recv(channel_receiver).unwrap();
    let mut hostVar: MemPtr = Default::default();
    hostVar.recv(channel_receiver).unwrap();
    let mut deviceName: Vec<u8> = Default::default();
    deviceName.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let mut dptr: CUdeviceptr = Default::default();
    let mut size: usize = Default::default();

    let module = get_module(fatCubinHandle).unwrap();
    let result = unsafe {
        cuModuleGetGlobal_v2(
            &mut dptr,
            &mut size,
            module,
            deviceName.as_ptr() as *const std::os::raw::c_char,
        )
    };
    add_variable(hostVar, dptr);

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}
