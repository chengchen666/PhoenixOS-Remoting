extern crate log;
mod dispatcher;

extern crate codegen;
extern crate cudasys;
extern crate network;

use codegen::gen_exe;
use cudasys::{
    cuda::{CUdeviceptr, CUfunction, CUmodule},
    cudart::{cudaDeviceSynchronize, cudaError_t, cudaGetDeviceCount, cudaSetDevice},
};
use dispatcher::dispatch;

#[allow(unused_imports)]
use network::{
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, RDMAChannelBufferManager,
    },
    type_impl::MemPtr,
    CommChannel, CommChannelError, Transportable,
    CONFIG,
};

#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, Level};

extern crate lazy_static;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;
use std::boxed::Box;

lazy_static! {
    // client_address -> module
    static ref MODULES: Mutex<HashMap<MemPtr, CUmodule>> = Mutex::new(HashMap::new());
    // host_func -> device_func
    static ref FUNCTIONS: Mutex<HashMap<MemPtr, CUfunction>> = Mutex::new(HashMap::new());
    // host_var -> device_var
    static ref VARIABLES: Mutex<HashMap<MemPtr, CUdeviceptr>> = Mutex::new(HashMap::new());
}

fn add_module(client_address: MemPtr, module: CUmodule) {
    MODULES.lock().unwrap().insert(client_address, module);
}

fn get_module(client_address: MemPtr) -> Option<CUmodule> {
    MODULES.lock().unwrap().get(&client_address).cloned()
}

fn add_function(host_func: MemPtr, device_func: CUfunction) {
    FUNCTIONS.lock().unwrap().insert(host_func, device_func);
}

fn get_function(host_func: MemPtr) -> Option<CUfunction> {
    FUNCTIONS.lock().unwrap().get(&host_func).cloned()
}

fn add_variable(host_var: MemPtr, device_var: CUdeviceptr) {
    VARIABLES.lock().unwrap().insert(host_var, device_var);
}

fn get_variable(host_var: MemPtr) -> Option<CUdeviceptr> {
    VARIABLES.lock().unwrap().get(&host_var).cloned()
}

fn create_buffer() -> (
    RingBuffer,
    RingBuffer,
) {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the server side, the sender's name is stoc_channel_name,
    // receiver's name is ctos_channel_name.
    match CONFIG.comm_type.as_str() {
        #[cfg(feature = "shm")]
        "shm" => {
            let sender = SHMChannelBufferManager::new_server(&CONFIG.stoc_channel_name, CONFIG.buf_size).unwrap();
            let receiver = SHMChannelBufferManager::new_server(&CONFIG.ctos_channel_name, CONFIG.buf_size).unwrap();
            (
                RingBuffer::new(Box::new(sender)),
                RingBuffer::new(Box::new(receiver)),
            )
        }
        #[cfg(feature = "rdma")]
        "rdma" => {
            let sender = RDMAChannelBufferManager::new_server(&CONFIG.stoc_channel_name, CONFIG.buf_size, CONFIG.sender_socket.parse().unwrap()).unwrap();
            let receiver = RDMAChannelBufferManager::new_server(&CONFIG.ctos_channel_name, CONFIG.buf_size, CONFIG.receiver_socket.parse().unwrap()).unwrap();
            (
                RingBuffer::new(Box::new(sender)),
                RingBuffer::new(Box::new(receiver)),
            )
        }
        &_ => panic!("Unsupported communication type in config"),
    }
}

fn receive_request<T: CommChannel>(channel_receiver: &mut T) -> Result<i32, CommChannelError> {
    let mut proc_id = 0;
    if let Ok(()) = proc_id.recv(channel_receiver) {
        Ok(proc_id)
    } else {
        Err(CommChannelError::IoError)
    }
}

pub fn launch_server() {
    let (mut channel_sender, mut channel_receiver) = create_buffer();
    info!("[{}:{}] shm buffer created", std::file!(), std::line!());
    let mut max_devices = 0;
    if let cudaError_t::cudaSuccess =
        unsafe { cudaGetDeviceCount(&mut max_devices as *mut ::std::os::raw::c_int) }
    {
        info!(
            "[{}:{}] found {} cuda devices",
            std::file!(),
            std::line!(),
            max_devices
        );
    } else {
        error!(
            "[{}:{}] failed to find cuda devices",
            std::file!(),
            std::line!()
        );
        panic!();
    }
    if let cudaError_t::cudaSuccess = unsafe { cudaSetDevice(0) } {
        info!("[{}:{}] cuda device set to 0", std::file!(), std::line!());
    } else {
        error!(
            "[{}:{}] failed to set cuda device",
            std::file!(),
            std::line!()
        );
        panic!();
    }
    if let cudaError_t::cudaSuccess = unsafe { cudaDeviceSynchronize() } {
        info!(
            "[{}:{}] cuda driver initialized",
            std::file!(),
            std::line!()
        );
    } else {
        error!(
            "[{}:{}] failed to initialize cuda driver",
            std::file!(),
            std::line!()
        );
        panic!();
    }

    loop {
        if let Ok(proc_id) = receive_request(&mut channel_receiver) {
            dispatch(proc_id, &mut channel_sender, &mut channel_receiver);
        } else {
            error!(
                "[{}:{}] failed to receive request",
                std::file!(),
                std::line!()
            );
            break;
        }
    }

    info!("[{}:{}] server terminated", std::file!(), std::line!());
}
