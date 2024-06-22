use crate::{RawMemory, Transportable};
use std::time::UNIX_EPOCH;
#[derive(Debug, Default, Clone)]
pub struct Request {
    pub timestamp: NsTimestamp,
    pub proc_id: i32,
    pub data: Vec<u8>,
}

impl Request {
    pub fn new(proc_id: i32, data: Vec<u8>) -> Request {
        Request {
            timestamp: NsTimestamp::new(),
            proc_id,
            data,
        }
    }
}

impl Transportable for Request {
    fn emulate_send<T: crate::CommChannel>(&self, channel: &mut T) -> Result<(), crate::CommChannelError> {
        self.timestamp.emulate_send(channel)?;
        self.proc_id.emulate_send(channel)?;
        self.data.emulate_send(channel)
    }
    fn send<T: crate::CommChannel>(
        &self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.timestamp.send(channel)?;
        self.proc_id.send(channel)?;
        self.data.send(channel)
    }

    fn recv<T: crate::CommChannel>(
        &mut self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.timestamp.recv(channel)?;
        self.proc_id.recv(channel)?;
        self.data.recv(channel)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NsTimestamp {
    pub sec_timestamp: i64,
    pub ns_timestamp: u32,
}
impl PartialEq for NsTimestamp {
    fn eq(&self, other: &Self) -> bool {
        self.sec_timestamp == other.sec_timestamp && self.ns_timestamp == other.ns_timestamp
    }
}

impl PartialOrd for NsTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.sec_timestamp.cmp(&other.sec_timestamp) {
            std::cmp::Ordering::Equal => self.ns_timestamp.partial_cmp(&other.ns_timestamp),
            other_order => Some(other_order),
        }
    }
}

impl NsTimestamp {
    pub fn new() -> NsTimestamp {
        NsTimestamp {
            sec_timestamp: 0,
            ns_timestamp: 0,
        }
    }
    pub fn now() -> NsTimestamp {
        let now_time = std::time::SystemTime::now();
        let duration_since_epoch = now_time.duration_since(UNIX_EPOCH).expect("Time went backwards");
        let sec = duration_since_epoch.as_secs() as i64;
        let ns = duration_since_epoch.subsec_nanos(); 
        NsTimestamp {
            sec_timestamp: sec,
            ns_timestamp: ns,
        }
    }
}

impl NsTimestamp {
    pub fn to_raw_memory(&self) -> RawMemory {
        let mut data = [0u8; std::mem::size_of::<NsTimestamp>()];
        // 0-7 bytes for sec_timestamp
        data[0..8].copy_from_slice(&self.sec_timestamp.to_le_bytes());
        // 8-11 bytes for ms_timestamp
        data[8..12].copy_from_slice(&self.ns_timestamp.to_le_bytes());
        RawMemory::from_ptr(data.as_ptr(), data.len())
    }
}

impl Transportable for NsTimestamp {
    fn emulate_send<T: crate::CommChannel>(&self, channel: &mut T) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.emulate_send(channel)?;
        self.ns_timestamp.emulate_send(channel)
    }
    fn send<T: crate::CommChannel>(
        &self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.send(channel)?;
        self.ns_timestamp.send(channel)
    }

    fn recv<T: crate::CommChannel>(
        &mut self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.recv(channel)?;
        self.ns_timestamp.recv(channel)
    }
}
