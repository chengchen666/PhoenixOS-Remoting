#![expect(non_snake_case)]

use super::*;
use cudasys::cuda::*;

pub fn __cudaRegisterFatBinaryExe<C: CommChannel>(
    server: &mut ServerWorker<C>,
) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
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
    server.modules.insert(client_address, module);

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn __cudaRegisterFunctionExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
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

    let module = *server.modules.get(&fatCubinHandle).unwrap();
    let result = unsafe {
        cuModuleGetFunction(
            &mut device_func,
            module,
            deviceName.as_ptr() as *const std::os::raw::c_char,
        )
    };
    server.functions.insert(hostFun, device_func);

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn __cudaRegisterVarExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
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

    let module = *server.modules.get(&fatCubinHandle).unwrap();
    let result = unsafe {
        cuModuleGetGlobal_v2(
            &mut dptr,
            &mut size,
            module,
            deviceName.as_ptr() as *const std::os::raw::c_char,
        )
    };
    server.variables.insert(hostVar, dptr);

    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}
