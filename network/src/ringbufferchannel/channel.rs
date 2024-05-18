use super::{ChannelBufferManager, CACHE_LINE_SZ};
use crate::{CommChannel, CommChannelError, RawMemory, RawMemoryMut};
use std::boxed::Box;

/// we reserve the first 128B for the header and tailer
/// 128 = cacheline size * 2
/// FIXME: should be hardware dependent
///
pub const HEAD_OFF: usize = 0;
pub const TAIL_OFF: usize = CACHE_LINE_SZ;
pub const META_AREA: usize = CACHE_LINE_SZ * 2;

/// A ring buffer where the buffer can be shared between different processes/threads
/// It uses the head 4B + 4B to store the head and tail
///
/// # Example
///
/// ```no_compile
/// use ringbufferchannel::{LocalChannelBufferManager, RingBuffer};
/// use crate::CommChannel;
/// use std::boxed::Box;
///
/// let mut buffer: RingBuffer = RingBuffer::new(Box::new(LocalChannelBufferManager::new(10 + 8)));
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
pub struct RingBuffer {
    manager: Box<dyn ChannelBufferManager>,
    capacity: usize, // Capacity of the buffer excluding head and tail.
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer
{
    pub fn new(manager: Box<dyn ChannelBufferManager>) -> RingBuffer {
        let (_ptr, len) = manager.get_managed_memory();
        assert!(len > META_AREA, "Buffer size is too small");
        // assert!(
        //     super::utils::is_cache_line_aligned(ptr),
        //     "Buffer is not cache line aligned"
        // );

        let capacity = len - META_AREA;
        let mut res = RingBuffer {
            manager,
            capacity,
        };
        res.write_head_volatile(0);
        res.write_tail_volatile(0);
        res
    }
}

impl CommChannel for RingBuffer {
    fn put_bytes(&mut self, src: &RawMemory) -> Result<usize, CommChannelError> {
        let mut len = src.len;
        let mut offset = 0;

        while len > 0 {
            // current head and tail
            let read_tail = self.read_tail_volatile() as usize;
            assert!(read_tail < self.capacity, "read_tail: {}", read_tail);

            // buf_head can be modified by the other side at any time
            // so we need to read it at the beginning and assume it is not changed
            if self.num_bytes_stored() == self.capacity {
                self.flush_out()?;
            }

            let current = std::cmp::min(self.num_adjacent_bytes_to_write(read_tail), len);

            unsafe {
                let _ = self.manager.write_at(META_AREA + read_tail, src.ptr.add(offset), current);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            self.write_tail_volatile(((read_tail + current) % self.capacity) as u32);
            offset += current;
            len -= current;
        }

        Ok(offset)
    }

    fn get_bytes(&mut self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let mut cur_recv = 0;
        while cur_recv != dst.len {
            let mut new_dst = dst.add_offset(cur_recv);
            let recv = self.try_get_bytes(&mut new_dst)?;
            cur_recv += recv;
        }
        Ok(cur_recv)
    }

    fn try_put_bytes(&mut self, _src: &RawMemory) -> Result<usize, CommChannelError> {
        unimplemented!()
    }

    fn try_get_bytes(&mut self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        let mut len = dst.len;
        let mut offset = 0;

        while len > 0 {
            if self.empty() {
                return Ok(offset);
            }

            let read_head = self.read_head_volatile() as usize;
            assert!(read_head < self.capacity, "read_head: {}", read_head);
            let current = std::cmp::min(self.num_adjacent_bytes_to_read(read_head), len);

            unsafe {
                self.manager.read_at(META_AREA + read_head, dst.ptr.add(offset), current);
            }

            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            assert!(
                read_head + current <= self.capacity,
                "read_head: {}, current: {}, capacity: {}",
                read_head,
                current,
                self.capacity
            );
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

impl RingBuffer
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

impl RingBuffer
{
    // WARNING: May need volatile in the future
    fn read_head_volatile(&self) -> u32 {
        let mut head: u32 = 0;
        self.manager.read_at(HEAD_OFF, &mut head as *mut u32 as *mut u8, std::mem::size_of::<u32>());
        head
    }

    fn write_head_volatile(&mut self, head: u32) {
        self.manager.write_at(HEAD_OFF, &head as *const u32 as *const u8, std::mem::size_of::<u32>());
    }

    fn read_tail_volatile(&self) -> u32 {
        let mut tail: u32 = 0;
        self.manager.read_at(TAIL_OFF, &mut tail as *mut u32 as *mut u8, std::mem::size_of::<u32>());
        tail
    }

    fn write_tail_volatile(&mut self, tail: u32) {
        self.manager.write_at(TAIL_OFF, &tail as *const u32 as *const u8, std::mem::size_of::<u32>());
    }

    #[inline]
    fn num_adjacent_bytes_to_read(&self, cur_head: usize) -> usize {
        let cur_tail = self.read_tail_volatile() as usize;
        if cur_tail >= cur_head {
            cur_tail - cur_head
        } else {
            self.capacity - cur_head
        }
    }

    #[inline]
    fn num_adjacent_bytes_to_write(&self, cur_tail: usize) -> usize {
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
