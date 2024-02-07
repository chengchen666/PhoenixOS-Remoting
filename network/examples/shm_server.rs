fn main() {
    #[cfg(target_os = "linux")]
    {
        use std::thread::sleep;
        use std::time::Duration;
        
        use network::{
            ringbufferchannel::{RingBuffer, SHMChannelBufferManager},
            CommChannel,
        };

        let shm_name = "/stoc";
        let shm_len = 1024;
        let manager = SHMChannelBufferManager::new_server(shm_name, shm_len).unwrap();
        let mut ring_buffer = RingBuffer::new(manager);

        loop {
            let mut dst = [0u8; 5];
            let res = ring_buffer.try_recv(&mut dst);
            match res {
                Ok(num) => {
                    if num > 0 {
                        println!("Received {:?}", dst);
                        break;
                    }

                    if num == 0 {
                        sleep(Duration::from_secs(1));
                    }
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
