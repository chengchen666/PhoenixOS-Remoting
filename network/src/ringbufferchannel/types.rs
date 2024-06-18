use crate::{RawMemory, Transportable};

#[derive(Debug, Default, Clone)]
pub struct Request {
    pub timestamp: UsTimestamp,
    pub proc_id: i32,
    pub data: Vec<u8>,
}

impl Request {
    pub fn new(proc_id: i32, data: Vec<u8>) -> Request {
        Request {
            timestamp: UsTimestamp::new(),
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
pub struct UsTimestamp {
    pub sec_timestamp: i64,
    pub us_timestamp: u32,
}
impl PartialEq for UsTimestamp {
    fn eq(&self, other: &Self) -> bool {
        self.sec_timestamp == other.sec_timestamp && self.us_timestamp == other.us_timestamp
    }
}

impl PartialOrd for UsTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.sec_timestamp.cmp(&other.sec_timestamp) {
            std::cmp::Ordering::Equal => self.us_timestamp.partial_cmp(&other.us_timestamp),
            other_order => Some(other_order),
        }
    }
}

impl UsTimestamp {
    pub fn new() -> UsTimestamp {
        UsTimestamp {
            sec_timestamp: 0,
            us_timestamp: 0,
        }
    }
    pub fn from_datetime(datetime: chrono::DateTime<chrono::Utc>) -> UsTimestamp {
        UsTimestamp {
            sec_timestamp: datetime.timestamp(),
            us_timestamp: datetime.timestamp_subsec_micros(),
        }
    }
}

impl UsTimestamp {
    pub fn to_raw_memory(&self) -> RawMemory {
        let mut data = [0u8; std::mem::size_of::<UsTimestamp>()];
        // 0-7 bytes for sec_timestamp
        data[0..8].copy_from_slice(&self.sec_timestamp.to_le_bytes());
        // 8-11 bytes for ms_timestamp
        data[8..12].copy_from_slice(&self.us_timestamp.to_le_bytes());
        RawMemory::from_ptr(data.as_ptr(), data.len())
    }
}

impl Transportable for UsTimestamp {
    fn emulate_send<T: crate::CommChannel>(&self, channel: &mut T) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.emulate_send(channel)?;
        self.us_timestamp.emulate_send(channel)
    }
    fn send<T: crate::CommChannel>(
        &self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.send(channel)?;
        self.us_timestamp.send(channel)
    }

    fn recv<T: crate::CommChannel>(
        &mut self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.recv(channel)?;
        self.us_timestamp.recv(channel)
    }
}
