use network::{
    ringbufferchannel::TcpChannel, CommChannel, CommChannelInnerIO, RawMemory, RawMemoryMut,
};

#[test]
fn tcp_channel_buffer_manager() {
    let addr = "127.0.0.1:8001";

    let s_handler =
        std::thread::spawn(
            move || match TcpChannel::new_server(addr) {
                Ok(server) => {
                    println!("Server created successfully");
                    server
                }
                Err(_) => {
                    panic!("Server creation failed");
                }
            },
        );

    // sender
    let c_handler =
        std::thread::spawn(
            move || match TcpChannel::new_client(addr, true) {
                Ok(client) => {
                    println!("Client created successfully");
                    client
                }
                Err(_) => {
                    panic!("Server creation failed");
                }
            },
        );

    let mut recver = s_handler.join().unwrap();
    recver.accept_connection().expect("Failed to accept client");

    let sender = c_handler.join().expect("Client thread panicked");

    // Send and receive data
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
