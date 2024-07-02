use std::sync::{Arc, Mutex};

use crate::{
    CONFIG, CommChannelInner, CommChannelInnerIO, CommChannelError,
    RawMemory, RawMemoryMut,
};
use super::types::*;

pub struct EmulatorChannel {
    manager: Box<dyn CommChannelInner>,
    byte_cnt: Arc<Mutex<usize>>,
    last_timestamp: Arc<Mutex<NsTimestamp>>,
    rtt: f64,
    bandwidth: f64,
    start: Arc<Mutex<Option<u64>>>,
    // begin: NsTimestamp,
}

unsafe impl Send for EmulatorChannel {}
unsafe impl Sync for EmulatorChannel {}

impl EmulatorChannel {
    pub fn new(manager: Box<dyn CommChannelInner>) -> Self {
        // let now = NsTimestamp::now();
        // log::info!("{}:{}", now.sec_timestamp, now.ns_timestamp);
        Self {
            manager,
            byte_cnt: Arc::new(Mutex::new(0)),
            last_timestamp: Arc::new(Mutex::new(NsTimestamp::new())),
            rtt: CONFIG.rtt,
            bandwidth: CONFIG.bandwidth,
            start: Arc::new(Mutex::new(None)),
            // begin: now,
        }
    }

    fn calculate_latency(&self, current_bytes: usize) -> f64 {
        let data_size =
            current_bytes + std::mem::size_of::<NsTimestamp>() + std::mem::size_of::<i32>();
        self.rtt * 1000000.0 / 2.0 + (data_size as f64 * 8.0 / self.bandwidth) * 1000000000.0 
    }

    pub fn calculate_ts(&self, current_bytes: usize) -> NsTimestamp {
        let latency = self.calculate_latency(current_bytes);
        let now_timestamp = NsTimestamp::now();
        let base_timestamp = match now_timestamp > self.get_last_timestamp() {
            true => now_timestamp,
            false => self.get_last_timestamp().clone(),
        };
        let sec = base_timestamp.sec_timestamp
            + (base_timestamp.ns_timestamp as i64 + latency as i64) / 1000000000;
        let ns = (base_timestamp.ns_timestamp + latency as u32) % 1000000000;
        NsTimestamp {
            sec_timestamp: sec,
            ns_timestamp: ns,
        }
    }

    fn send<T>(&self, src: T) -> Result<(), CommChannelError> {
        if self.get_start() == None {
            self.set_start(Some(measure::rdtscp()));
        }
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
    pub fn get_last_timestamp(&self) -> NsTimestamp {
        *self.last_timestamp.lock().unwrap()
    }

    #[inline]
    pub fn set_last_timestamp(&self, last_timestamp: NsTimestamp) {
        *self.last_timestamp.lock().unwrap() = last_timestamp;
    }

    #[inline]
    pub fn get_start(&self) -> Option<u64> {
        *self.start.lock().unwrap()
    }

    #[inline]
    pub fn set_start(&self, start: Option<u64>) {
        *self.start.lock().unwrap() = start;
    }
}

impl CommChannelInnerIO for EmulatorChannel {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        if self.get_start() == None {
            self.set_start(Some(measure::rdtscp()));
            // let now = NsTimestamp::now();
            // let elapsed = (now.sec_timestamp - self.begin.sec_timestamp) * 1000000000
            //     + (now.ns_timestamp as i32 - self.begin.ns_timestamp as i32) as i64;
            // log::info!(", {}", elapsed);
        }
        self.set_byte_cnt(self.get_byte_cnt() + src.len);
        self.manager.put_bytes(src)
    }

    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        if self.get_start() == None {
            self.set_start(Some(measure::rdtscp()));
        }
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
        if self.get_start() == None {
            self.set_start(Some(measure::rdtscp()));
        }
        let end = measure::rdtscp();
        let elapsed = measure::clock2ns(end - self.get_start().unwrap());
        #[cfg(not(feature = "log_server"))]
        log::info!(", {}", elapsed / 1000.0);
        let ts = self.calculate_ts(self.get_byte_cnt());
        let byte_cnt = self.get_byte_cnt();
        #[cfg(not(feature = "log_server"))]
        log::info!(", {}", byte_cnt);
        let _ = self.send(ts);
        let _ = self.manager.flush_out();
        self.set_byte_cnt(0);
        self.set_last_timestamp(ts);
        self.set_start(None);
        Ok(())
    }

    fn recv_ts(&self) -> Result<(), crate::CommChannelError> {
        let mut timestamp: NsTimestamp = Default::default();
        let _ = self.recv(&mut timestamp);
        while NsTimestamp::now() < timestamp {
            // Busy-waiting
        }
        // let start = NsTimestamp::now();
        // log::info!("gpu_issue, {}:{}", start.sec_timestamp, start.ns_timestamp as f64 / 1000.0);
        Ok(())
    }
}
