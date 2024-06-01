#[allow(non_snake_case)]
extern crate network;

use network::{
    ringbufferchannel::RDMAChannel, Channel, CommChannel, RawMemory, RawMemoryMut,
};
use std::net::SocketAddr;

const BUF_SIZE: usize = 1024 + network::ringbufferchannel::META_AREA;
const PORT: u8 = 1;

#[test]
fn rdma_channel_buffer_manager() {
    let name = "/ctos";

    let addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();

    // First, new a RDMA server to listen at a socket address (s_sender_addr).
    // Then new a client with server's socket address to handshake with it.
    // The client side will use the server name (s_sender_name) to get its
    // remote info like raddr and rkey.

    let s_handler =
        std::thread::spawn(
            move || match RDMAChannel::new_server(name, BUF_SIZE, addr) {
                Ok(server) => {
                    println!("Server created successfully");
                    server
                }
                Err(_) => {
                    panic!("Server creation failed");
                }
            },
        );

    let c_handler =
        std::thread::spawn(
            move || match RDMAChannel::new_client(name, BUF_SIZE, addr, PORT) {
                Ok(client) => {
                    println!("Client created successfully");
                    client
                }
                Err(_) => {
                    panic!("Server creation failed");
                }
            },
        );

    let r = s_handler.join().unwrap();
    let s = c_handler.join().unwrap();
    let recver = Channel::new(Box::new(r));
    let sender = Channel::new(Box::new(s));

    const SZ: usize = 256;
    let data = [48 as u8; SZ];
    let send_memory = RawMemory::new(&data, SZ);
    sender.put_bytes(&send_memory).unwrap();

    let data2 = [97 as u8; SZ];
    let send_memory = RawMemory::new(&data2, SZ);
    sender.put_bytes(&send_memory).unwrap();

    let _ = sender.flush_out();

    let mut buffer = [0u8; 2 * SZ];
    let mut recv_memory = RawMemoryMut::new(&mut buffer, 2 * SZ);
    match recver.get_bytes(&mut recv_memory) {
        Ok(size) => {
            for i in 0..SZ {
                assert_eq!(buffer[i], 48);
            }
            for i in SZ..size {
                assert_eq!(buffer[i], 97);
            }
        }
        Err(_) => todo!(),
    }
}
