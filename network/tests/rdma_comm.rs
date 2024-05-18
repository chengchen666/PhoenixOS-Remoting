#[allow(non_snake_case)]

extern crate network;

use std::net::SocketAddr;
use std::boxed::Box;
use network::{
    ringbufferchannel::{RDMAChannelBufferManager, RingBuffer},
    CommChannel, RawMemory, RawMemoryMut,
};

const BUF_SIZE: usize = 1024 + network::ringbufferchannel::channel::META_AREA;
const PORT: u8 = 1;

#[test]
fn rdma_channel_buffer_manager() {
    let s_sender_name = "/stoc";
    let c_sender_name = "/ctos";

    let s_sender_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
    let c_sender_addr: SocketAddr = "127.0.0.1:8002".parse().unwrap();

    // First, new a RDMA server to listen at a socket address (s_sender_addr).
    // Then new a client with server's socket address to handshake with it.
    // The client side will use the server name (s_sender_name) to get its
    // remote info like raddr and rkey.
    let s1 = RDMAChannelBufferManager::new_server(s_sender_name, BUF_SIZE, s_sender_addr).unwrap();
    let c1 = RDMAChannelBufferManager::new_client(s_sender_name, BUF_SIZE, s_sender_addr, PORT).unwrap();
    let mut s_sender = RingBuffer::new(Box::new(s1));
    let mut c_recver = RingBuffer::new(Box::new(c1));

    let s2 = RDMAChannelBufferManager::new_server(c_sender_name, BUF_SIZE, c_sender_addr).unwrap();
    let c2 = RDMAChannelBufferManager::new_client(c_sender_name, BUF_SIZE, c_sender_addr, PORT).unwrap();
    let mut s_recver = RingBuffer::new(Box::new(s2));
    let mut c_sender = RingBuffer::new(Box::new(c2));

    const SZ: usize = 20;
    let data = [48 as u8; SZ];
    let send_memory = RawMemory::new(&data, SZ);
    c_sender.put_bytes(&send_memory).unwrap();

    let mut buffer = [0u8; SZ];
    let mut recv_memory = RawMemoryMut::new(&mut buffer, SZ);
    match s_recver.get_bytes(&mut recv_memory) {
        Ok(size) => {
            for i in 0..size {
                assert_eq!(buffer[i], 48);
            }
        },
        Err(_) => todo!()
    }

    const SZ2: usize = 256;
    let data = [97 as u8; SZ2];
    let send_memory = RawMemory::new(&data, SZ2);
    s_sender.put_bytes(&send_memory).unwrap();

    let mut buffer = [0u8; SZ2];
    let mut recv_memory = RawMemoryMut::new(&mut buffer, SZ2);
    match c_recver.get_bytes(&mut recv_memory) {
        Ok(size) => {
            for i in 0..size {
                assert_eq!(buffer[i], 97);
            }
        },
        Err(_) => todo!()
    }
}
