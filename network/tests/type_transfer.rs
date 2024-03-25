extern crate cudasys;
extern crate network;

use cudasys::cudart::cudaError_t;
use network::{
    ringbufferchannel::{ChannelBufferManager, LocalChannelBufferManager, RingBuffer},
    CommChannel, FromPrimitive, Transportable,
};

use std::sync::{Arc, Barrier};
use std::thread;

pub struct ConsumerManager {
    buf: *mut u8,
    capacity: usize,
}

impl ChannelBufferManager for ConsumerManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.buf, self.capacity)
    }
}

impl ConsumerManager {
    pub fn new(producer: &LocalChannelBufferManager) -> Self {
        let (buf, capacity) = producer.get_managed_memory();
        ConsumerManager { buf, capacity }
    }
}

unsafe impl Send for ConsumerManager {}

#[test]
fn test_cudaerror() {
    let c_shared_buffer =
        LocalChannelBufferManager::new(1024 + network::ringbufferchannel::channel::META_AREA);
    let p_shared_buffer = ConsumerManager::new(&c_shared_buffer);

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        let mut producer_ring_buffer = RingBuffer::new(p_shared_buffer);
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let var = match cudaError_t::from_u32(i % 10) {
                Some(v) => v,
                None => panic!("failed to convert from u32"),
            };
            var.send(&mut producer_ring_buffer).unwrap();
            producer_ring_buffer.flush_out().unwrap();
        }

        println!("Producer done");
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        let mut consumer_ring_buffer = RingBuffer::new(c_shared_buffer);
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;

        while received < test_iters {
            let test = match cudaError_t::from_u32(received % 10) {
                Some(v) => v,
                None => panic!("failed to convert from u32"),
            };
            let mut var = cudaError_t::cudaSuccess;
            var.recv(&mut consumer_ring_buffer).unwrap();
            assert_eq!(var, test);
            received += 1;
        }
    });

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
