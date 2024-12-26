#![feature(maybe_uninit_slice)]

mod dispatcher;

use cudasys::{
    cuda::CUmodule,
    cudart::{cudaDeviceSynchronize, cudaError_t, cudaGetDeviceCount, cudaSetDevice},
};
use dispatcher::dispatch;

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::{
    ringbufferchannel::{EmulatorChannel, SHMChannel},
    Channel, CommChannel, CommChannelError, Transportable, NetworkConfig,
};

use log::{error, info};

#[cfg(feature = "shadow_desc")]
use std::collections::BTreeMap;

struct ServerWorker<C> {
    pub id: i32,
    pub channel_sender: C,
    pub channel_receiver: C,
    pub modules: Vec<CUmodule>,
    #[cfg(feature = "shadow_desc")]
    pub resources: BTreeMap<usize, usize>,
}

impl<C> Drop for ServerWorker<C> {
    fn drop(&mut self) {
        for module in &self.modules {
            unsafe {
                cudasys::cuda::cuModuleUnload(*module);
            }
        }
    }
}

fn create_buffer(#[expect(non_snake_case)] CONFIG: &NetworkConfig, id: i32) -> (Channel, Channel) {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the server side, the sender's name is stoc_channel_name,
    // receiver's name is ctos_channel_name.
    match CONFIG.comm_type.as_str() {
        "shm" => {
            let sender = SHMChannel::new_server_with_id(&CONFIG.stoc_channel_name, id, CONFIG.buf_size).unwrap();
            let receiver = SHMChannel::new_server_with_id(&CONFIG.ctos_channel_name, id, CONFIG.buf_size).unwrap();
            if cfg!(feature = "emulator") {
                return (
                    Channel::new(Box::new(EmulatorChannel::new(Box::new(sender)))),
                    Channel::new(Box::new(EmulatorChannel::new(Box::new(receiver)))),
                );
            }
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        #[cfg(feature = "rdma")]
        "rdma" => {
            // Make sure to new receiver first! Client side sender will handshake with it first.
            let receiver = RDMAChannel::new_server(
                &CONFIG.ctos_channel_name,
                CONFIG.buf_size,
                CONFIG.receiver_socket.parse().unwrap(),
            ).unwrap();
            let sender = RDMAChannel::new_server(
                &CONFIG.stoc_channel_name,
                CONFIG.buf_size,
                CONFIG.sender_socket.parse().unwrap(),
            ).unwrap();
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
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

pub fn launch_server(#[expect(non_snake_case)] CONFIG: &NetworkConfig, id: i32, tcp: Option<std::net::TcpStream>) {
    let (channel_sender, channel_receiver) = create_buffer(CONFIG, id);
    info!(
        "[{}:{}] {} buffer created",
        std::file!(),
        std::line!(),
        CONFIG.comm_type
    );
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

    let mut server = ServerWorker {
        id,
        channel_sender,
        channel_receiver,
        modules: Default::default(),
        #[cfg(feature = "shadow_desc")]
        resources: Default::default(),
    };

    if let Some(mut stream) = tcp {
        use std::io::Write as _;
        stream.write_all(&id.to_be_bytes()).unwrap();
    }

    loop {
        if let Ok(proc_id) = receive_request(&mut server.channel_receiver) {
            if proc_id == -1 {
                break;
            }
            dispatch(proc_id, &mut server);
        } else {
            error!(
                "[{}:{}] failed to receive request",
                std::file!(),
                std::line!()
            );
            break;
        }
    }

    info!("[{}:{}] server #{} terminated", std::file!(), std::line!(), server.id);
}
