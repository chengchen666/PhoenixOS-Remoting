fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::{
            ringbufferchannel::{RingBuffer, SHMChannelBufferManager},
            CommChannel,
        };

        let shm_name = "/stoc";
        let shm_len = 1024;
        let manager = SHMChannelBufferManager::new_client(shm_name, shm_len).unwrap();
        let mut ring_buffer = RingBuffer::new(manager);
        ring_buffer.send(&[1, 2, 3, 4, 5]).unwrap();

        println!("send done");
    }

    println!("SHM client only works on linux");
}
