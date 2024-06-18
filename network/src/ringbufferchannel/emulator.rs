use chrono::Utc;
use std::sync::{Arc, Mutex};

use crate::{
    CONFIG, CommChannelInner, CommChannelInnerIO, CommChannelError,
    RawMemory, RawMemoryMut,
};
use super::types::*;

pub struct EmulatorChannel {
    manager: Box<dyn CommChannelInner>,
    byte_cnt: Arc<Mutex<usize>>,
    last_timestamp: Arc<Mutex<UsTimestamp>>,
    rtt: f64,
    bandwidth: f64,
}

unsafe impl Send for EmulatorChannel {}
unsafe impl Sync for EmulatorChannel {}

impl EmulatorChannel {
    pub fn new(manager: Box<dyn CommChannelInner>) -> Self {
        Self {
            manager,
            byte_cnt: Arc::new(Mutex::new(0)),
            last_timestamp: Arc::new(Mutex::new(UsTimestamp::new())),
            rtt: CONFIG.rtt,
            bandwidth: CONFIG.bandwidth,
        }
    }

    fn calculate_latency(&self, current_bytes: usize) -> f64 {
        let data_size =
            current_bytes + std::mem::size_of::<UsTimestamp>() + std::mem::size_of::<i32>();
        self.rtt * 1000.0 / 2.0 + (data_size as f64 / self.bandwidth) * 1000000.0 * 8.0
    }

    pub fn calculate_ts(&self, current_bytes: usize) -> UsTimestamp {
        let latency = self.calculate_latency(current_bytes);
        let now_timestamp = UsTimestamp::from_datetime(Utc::now());
        let base_timestamp = match now_timestamp > self.get_last_timestamp() {
            true => now_timestamp,
            false => self.get_last_timestamp().clone(),
        };
        let sec = base_timestamp.sec_timestamp
            + (base_timestamp.us_timestamp as i64 + latency as i64) / 1000000;
        let us = (base_timestamp.us_timestamp + latency as u32) % 1000000;
        UsTimestamp {
            sec_timestamp: sec,
            us_timestamp: us,
        }
    }

    fn send<T>(&self, src: T) -> Result<(), CommChannelError> {
        let memory = RawMemory::new(&src, std::mem::size_of::<T>());
        match self.manager.put_bytes(&memory)? == std::mem::size_of::<T>() {
            true => {
                Ok(())},
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<T>(&self, dst: &mut T) -> Result<(), CommChannelError> {
        let mut memory = RawMemoryMut::new(dst, std::mem::size_of::<T>());
        match self.manager.get_bytes(&mut memory)? == std::mem::size_of::<T>() {
            true => {Ok(())},
            false => Err(CommChannelError::IoError),
        }
    }

    #[inline]
    pub fn get_byte_cnt(&self) -> usize {
        *self.byte_cnt.lock().unwrap()
    }

    #[inline]
    pub fn set_byte_cnt(&self, byte_cnt: usize) {
        *self.byte_cnt.lock().unwrap() = byte_cnt;
    }

    #[inline]
    pub fn get_last_timestamp(&self) -> UsTimestamp {
        *self.last_timestamp.lock().unwrap()
    }

    #[inline]
    pub fn set_last_timestamp(&self, last_timestamp: UsTimestamp) {
        *self.last_timestamp.lock().unwrap() = last_timestamp;
    }
}

impl CommChannelInnerIO for EmulatorChannel {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.set_byte_cnt(self.get_byte_cnt() + src.len);
        self.manager.put_bytes(src)
    }

    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.manager.try_put_bytes(src)
    }

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.manager.get_bytes(dst)
    }

    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.manager.try_get_bytes(dst)
    }

    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.manager.safe_try_get_bytes(dst)
    }
}

impl CommChannelInner for EmulatorChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        let _ = self.manager.flush_out();
        let ts = self.calculate_ts(self.get_byte_cnt());
        let _ = self.send(ts);
        self.set_byte_cnt(0);
        self.set_last_timestamp(ts);
        Ok(())
    }

    fn recv_ts(&self) -> Result<(), crate::CommChannelError> {
        let mut timestamp: UsTimestamp = Default::default();
        let _ = self.recv(&mut timestamp);
        while UsTimestamp::from_datetime(Utc::now()) < timestamp {
            // Busy-waiting
        }
        Ok(())
    }
}
