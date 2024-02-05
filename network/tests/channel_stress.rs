extern crate network;

use network::{
    ringbufferchannel::{ChannelBufferManager, LocalChannelBufferManager, RingBuffer},
    CommChannel,
};

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

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
fn test_ring_buffer_producer_consumer() {
    let p_shared_buffer = LocalChannelBufferManager::new(1024 + 8);
    let c_shared_buffer = ConsumerManager::new(&p_shared_buffer);

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        let mut producer_ring_buffer = RingBuffer::new(p_shared_buffer);
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let data = [(i % 256) as u8; 10]; // Simplified data to send
            producer_ring_buffer.send(&data).unwrap();
        }
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        let mut consumer_ring_buffer = RingBuffer::new(c_shared_buffer);
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;
        let mut buffer = [0u8; 10];

        while received < test_iters {
            match consumer_ring_buffer.recv(&mut buffer) {
                Ok(size) => {
                    for i in 0..size {
                        assert_eq!(buffer[i], (received % 256) as u8);
                    }

                    received += 1;
                }
                Err(_) => thread::sleep(Duration::from_millis(10)), // Wait if buffer is empty
            }
        }
    }); 

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
