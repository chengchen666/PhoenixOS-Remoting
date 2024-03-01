fn main() {
    #[cfg(target_os = "linux")]
    {
        use network::{
            ringbufferchannel::{RingBuffer, SHMChannelBufferManager},
            Transportable
        };

        let shm_name = "/stoc";
        let shm_len = 1024;
        let manager = SHMChannelBufferManager::new_client(shm_name, shm_len).unwrap();
        let mut ring_buffer = RingBuffer::new(manager);
        let buf = [1, 2, 3, 4, 5];
        buf.send(&mut ring_buffer).unwrap();

        println!("send done");
    }

    println!("SHM client only works on linux");
}
