use chrono::Utc;

use crate::{CommChannel, Transportable, CONFIG};

use super::{types::*, ChannelBufferManager, CACHE_LINE_SZ};

const HEAD_OFF: usize = 0;
const TAIL_OFF: usize = CACHE_LINE_SZ;
const META_AREA: usize = CACHE_LINE_SZ * 2;
pub struct EmulatorBuffer {
    manager: Box<dyn ChannelBufferManager>,
    capacity: usize,
    byte_cnt: usize,
    last_timestamp: MsTimestamp,
    rtt: f64,
    bandwidth: f64,
}

unsafe impl Send for EmulatorBuffer {}
unsafe impl Sync for EmulatorBuffer {}

impl EmulatorBuffer {
    pub fn new(manager: Box<dyn ChannelBufferManager>) -> EmulatorBuffer {
        let (_ptr, len) = manager.get_managed_memory();
        assert!(len > META_AREA, "Buffer size is too small");

        let capacity = len - META_AREA;
        let mut res = EmulatorBuffer {
            manager,
            capacity,
            byte_cnt: 0,
            last_timestamp: MsTimestamp::new(),
            rtt: CONFIG.rtt,
            bandwidth: CONFIG.bandwidth,
        };
        res.write_head_volatile(0);
        res.write_tail_volatile(0);
        res
    }
    fn calculate_latency(&self, current_bytes: usize) -> f64 {
        let data_size =
            current_bytes + std::mem::size_of::<MsTimestamp>() + std::mem::size_of::<i32>();
        self.rtt / 2.0 + (data_size as f64 / self.bandwidth) * 1000.0 * 8.0
    }
    pub fn calculate_ts(&self, current_bytes: usize) -> MsTimestamp {
        let latency = self.calculate_latency(current_bytes);
        let now = Utc::now();
        let now_timestamp = MsTimestamp {
            sec_timestamp: now.timestamp(),
            ms_timestamp: now.timestamp_subsec_millis(),
        };
        let base_timestamp = match now_timestamp > self.last_timestamp {
            true => now_timestamp,
            false => self.last_timestamp.clone(),
        };
        let sec = base_timestamp.sec_timestamp
            + (base_timestamp.ms_timestamp as i64 + latency as i64) / 1000;
        let ms = (base_timestamp.ms_timestamp + latency as u32) % 1000;
        MsTimestamp {
            sec_timestamp: sec,
            ms_timestamp: ms,
        }
    }
}

impl CommChannel for EmulatorBuffer {
    fn put_bytes(&mut self, src: &crate::RawMemory) -> Result<usize, crate::CommChannelError> {
        let mut len = src.len;
        let mut offset = 0;
        self.byte_cnt += len;

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
                let _ = self
                    .manager
                    .write_at(META_AREA + read_tail, src.ptr.add(offset), current);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            self.write_tail_volatile(((read_tail + current) % self.capacity) as u32);
            offset += current;
            len -= current;
        }
        Ok(offset)
    }

    fn try_put_bytes(&mut self, _src: &crate::RawMemory) -> Result<usize, crate::CommChannelError> {
        unimplemented!();
    }

    fn get_bytes(
        &mut self,
        dst: &mut crate::RawMemoryMut,
    ) -> Result<usize, crate::CommChannelError> {
        let mut cur_recv = 0;
        while cur_recv != dst.len {
            let mut new_dst = dst.add_offset(cur_recv);
            let recv = self.try_get_bytes(&mut new_dst)?;
            cur_recv += recv;
        }
        Ok(cur_recv)
    }

    fn try_get_bytes(
        &mut self,
        dst: &mut crate::RawMemoryMut,
    ) -> Result<usize, crate::CommChannelError> {
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
                self.manager
                    .read_at(META_AREA + read_head, dst.ptr.add(offset), current);
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

    fn safe_try_get_bytes(
        &mut self,
        dst: &mut crate::RawMemoryMut,
    ) -> Result<usize, crate::CommChannelError> {
        if self.num_bytes_stored() < dst.len {
            return Ok(0);
        } else {
            self.try_get_bytes(dst)
        }
    }

    fn flush_out(&mut self) -> Result<(), crate::CommChannelError> {
        while self.num_bytes_stored() == self.capacity {
            // Busy-waiting
        }
        let ts = self.calculate_ts(self.byte_cnt);
        ts.send(self)?;
        self.last_timestamp = ts;
        self.byte_cnt = 0;
        Ok(())
    }

    fn recv_ts(&mut self) -> Result<(), crate::CommChannelError> {
        let mut timestamp: MsTimestamp = Default::default();
        let _ = timestamp.recv(self);
        while MsTimestamp::from_datetime(Utc::now()) < timestamp {
            // Busy-waiting
        }
        Ok(())
    }
}

impl EmulatorBuffer {
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

impl EmulatorBuffer {
    // WARNING: May need volatile in the future
    fn read_head_volatile(&self) -> u32 {
        let mut head: u32 = 0;
        self.manager.read_at(
            HEAD_OFF,
            &mut head as *mut u32 as *mut u8,
            std::mem::size_of::<u32>(),
        );
        head
    }

    fn write_head_volatile(&mut self, head: u32) {
        self.manager.write_at(
            HEAD_OFF,
            &head as *const u32 as *const u8,
            std::mem::size_of::<u32>(),
        );
    }

    fn read_tail_volatile(&self) -> u32 {
        let mut tail: u32 = 0;
        self.manager.read_at(
            TAIL_OFF,
            &mut tail as *mut u32 as *mut u8,
            std::mem::size_of::<u32>(),
        );
        tail
    }

    fn write_tail_volatile(&mut self, tail: u32) {
        self.manager.write_at(
            TAIL_OFF,
            &tail as *const u32 as *const u8,
            std::mem::size_of::<u32>(),
        );
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
