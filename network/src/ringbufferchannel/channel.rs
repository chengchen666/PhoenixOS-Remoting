use std::ptr::{self, NonNull};

use super::ChannelBufferManager;
use crate::{CommChannel, CommChannelError};

/// A ring buffer where the buffer can be shared between different processes/threads
/// It uses the head 4B + 4B to store the head and tail
///
/// # Example
///
/// ```no_compile
/// use ringbufferchannel::{LocalChannelBufferManager, RingBuffer};
/// use crate::CommChannel;
///
/// let mut buffer: RingBuffer<LocalChannelBufferManager> = RingBuffer::new(LocalChannelBufferManager::new(10 + 8));
/// let data_to_send = [1, 2, 3, 4, 5];
/// let mut receive_buffer = [0u8; 5];
///
/// buffer.send(&data_to_send).unwrap();
/// buffer.recv(&mut receive_buffer).unwrap();
///
/// assert_eq!(receive_buffer, data_to_send);
///
/// ```
///
pub struct RingBuffer<T: ChannelBufferManager> {
    _manager: T,
    buffer: NonNull<u8>,
    capacity: usize, // Capacity of the buffer excluding head and tail.
}

impl<T> RingBuffer<T>
where
    T: ChannelBufferManager,
{
    pub fn new(manager: T) -> RingBuffer<T> {
        let (ptr, len) = manager.get_managed_memory();
        assert!(
            len >= std::mem::size_of::<u32>() * 2,
            "Buffer size is too small"
        );
        let buffer: NonNull<u8> = NonNull::new(ptr).unwrap();

        let capacity = len - std::mem::size_of::<u32>() * 2;
        let mut res = RingBuffer {
            _manager: manager,
            buffer,
            capacity,
        };
        res.write_head_volatile(0);
        res.write_tail_volatile(0);
        res
    }
}

impl<T: ChannelBufferManager> CommChannel for RingBuffer<T> {
    fn send(&mut self, src: &[u8]) -> Result<usize, CommChannelError> {
        let mut len = src.len();
        let mut offset = 0;

        while len > 0 {
            // current head and tail
            let read_tail = self.read_tail_volatile() as usize;

            // buf_head can be modified by the other side at any time
            // so we need to read it at the beginning and assume it is not changed
            if self.num_bytes_stored() == self.capacity { 
                self.flush_out()?;
            }

            let current = std::cmp::min(self.num_adjacent_bytes_to_write(read_tail), len);
            
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(offset),
                    self.buffer
                        .as_ptr()
                        .add(read_tail + std::mem::size_of::<u32>() * 2),
                    current,
                );
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            self.write_tail_volatile(((read_tail + current) % self.capacity) as u32);
            offset += current;
            len -= current;
        }

        Ok(src.len())
    }

    fn recv(&mut self, dst: &mut [u8]) -> Result<usize, CommChannelError> {
        let mut cur_recv = 0;
        while cur_recv != dst.len() {
            let new_dst = &mut dst[cur_recv..];
            let recv = self.try_recv(new_dst)?;
            cur_recv += recv;
        }
        Ok(cur_recv)
    }

    fn try_send(&mut self, _src: &[u8]) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn try_recv(&mut self, dst: &mut [u8]) -> Result<usize, CommChannelError> {
        let mut len = dst.len();
        let mut offset = 0;

        while len > 0 {
            if self.empty() {
                return Ok(offset);
            }

            let read_head = self.read_head_volatile() as usize;
            let current = std::cmp::min(self.num_adjacent_bytes_to_read(read_head), len);

            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buffer
                        .as_ptr()
                        .add(read_head + std::mem::size_of::<u32>() * 2),
                    dst.as_mut_ptr().add(offset),
                    current,
                );
            }

            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            self.write_head_volatile(((read_head + current) % self.capacity) as u32);
            offset += current;
            len -= current;
        }

        Ok(offset)
    }

    fn flush_out(&mut self) -> Result<(), CommChannelError> {
        while self.num_bytes_stored() == self.capacity {
            // Busy-waiting
        }
        Ok(())
    }
}

impl<T> RingBuffer<T>
where
    T: ChannelBufferManager,
{
    /// The space that has not been consumed by the consumer
    #[inline]
    pub fn num_bytes_free(&self) -> usize {
        self.capacity - self.num_bytes_stored()
    }

    #[inline]
    pub fn num_bytes_stored(&self) -> usize {
        let head = self.read_head_volatile() as usize;
        let tail = self.read_tail_volatile() as usize;

        if tail >= head {
            // Tail is ahead of head
            tail - head
        } else {
            // Head is ahead of tail, buffer is wrapped
            self.capacity - (head - tail)
        }        
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.read_head_volatile() == self.read_tail_volatile()
    }
}

impl<T> RingBuffer<T>
where
    T: ChannelBufferManager,
{
    fn read_head_volatile(&self) -> u32 {
        unsafe { ptr::read_volatile(self.buffer.as_ptr() as *const u32) }
    }

    fn write_head_volatile(&mut self, head: u32) {
        unsafe {
            ptr::write_volatile(self.buffer.as_ptr() as *mut u32, head);
        }
    }

    fn read_tail_volatile(&self) -> u32 {
        unsafe { ptr::read_volatile((self.buffer.as_ptr() as *const u32).add(1)) }
    }

    fn write_tail_volatile(&mut self, tail: u32) {
        unsafe {
            ptr::write_volatile((self.buffer.as_ptr() as *mut u32).add(1), tail);
        }
    }

    #[inline]
    fn num_adjacent_bytes_to_read(&self, cur_head : usize) -> usize {
        let cur_tail = self.read_tail_volatile() as usize;
        if cur_tail >= cur_head {
            cur_tail - cur_head 
        } else {
            self.capacity - cur_head 
        }
    }

    #[inline]
    fn num_adjacent_bytes_to_write(&self, cur_tail : usize) -> usize { 
        let mut cur_head = self.read_head_volatile() as usize;        
        if cur_head == 0 {
            cur_head = self.capacity;
        }

        if cur_tail >= cur_head {
            self.capacity - cur_tail
        } else {
            cur_head - cur_tail - 1
        }
    }
    
}
