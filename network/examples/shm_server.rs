fn main() {
    #[cfg(target_os = "linux")]
    {   
        use network::{
            ringbufferchannel::{RingBuffer, SHMChannelBufferManager},
            Transportable
        };

        let shm_name = "/stoc";
        let shm_len = 1024;
        let manager = SHMChannelBufferManager::new_server(shm_name, shm_len).unwrap();
        let mut ring_buffer = RingBuffer::new(manager);

        loop {
            let mut dst = [0u8; 5];
            let res = dst.recv(&mut ring_buffer);
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
