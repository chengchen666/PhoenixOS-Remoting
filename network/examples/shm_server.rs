fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::{ringbufferchannel::SHMChannel, Channel, Transportable};
        use std::boxed::Box;

        let shm_name = "/stoc";
        let shm_len = 1024;
        let channel = Channel::new(Box::new(SHMChannel::new_server(shm_name, shm_len).unwrap()));

        loop {
            let mut dst = [0u8; 5];
            let res = dst.recv(&channel);
            match res {
                Ok(()) => {
                    println!("Received {:?}", dst);
                    break;
                }
                Err(e) => {
                    println!("Error {}", e);
                    assert!(false);
                }
            }
        }
    }

    println!("SHM server only works on linux");
}
