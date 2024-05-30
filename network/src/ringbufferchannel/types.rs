use log::info;

use crate::{RawMemory, Transportable};

#[derive(Debug, Default, Clone)]
pub struct Request {
    pub timestamp: MsTimestamp,
    pub proc_id: i32,
    pub data: Vec<u8>,
}

impl Request {
    pub fn new(proc_id: i32, data: Vec<u8>) -> Request {
        Request {
            timestamp: MsTimestamp::new(),
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

    fn try_recv<T: crate::CommChannel>(&mut self, channel: &mut T) -> Result<(), crate::CommChannelError> {
        self.timestamp.try_recv(channel)?;
        self.proc_id.recv(channel)?;
        self.data.recv(channel)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MsTimestamp {
    pub sec_timestamp: i64,
    pub ms_timestamp: u32,
}
impl PartialEq for MsTimestamp {
    fn eq(&self, other: &Self) -> bool {
        self.sec_timestamp == other.sec_timestamp && self.ms_timestamp == other.ms_timestamp
    }
}

impl PartialOrd for MsTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.sec_timestamp.cmp(&other.sec_timestamp) {
            std::cmp::Ordering::Equal => self.ms_timestamp.partial_cmp(&other.ms_timestamp),
            other_order => Some(other_order),
        }
    }
}

impl MsTimestamp {
    pub fn new() -> MsTimestamp {
        MsTimestamp {
            sec_timestamp: 0,
            ms_timestamp: 0,
        }
    }
    pub fn from_datetime(datetime: chrono::DateTime<chrono::Utc>) -> MsTimestamp {
        MsTimestamp {
            sec_timestamp: datetime.timestamp(),
            ms_timestamp: datetime.timestamp_subsec_millis(),
        }
    }
}

impl MsTimestamp {
    pub fn to_raw_memory(&self) -> RawMemory {
        let mut data = [0u8; std::mem::size_of::<MsTimestamp>()];
        // 0-7 bytes for sec_timestamp
        data[0..8].copy_from_slice(&self.sec_timestamp.to_le_bytes());
        // 8-11 bytes for ms_timestamp
        data[8..12].copy_from_slice(&self.ms_timestamp.to_le_bytes());
        RawMemory::from_ptr(data.as_ptr(), data.len())
    }
}

impl Transportable for MsTimestamp {
    fn emulate_send<T: crate::CommChannel>(&self, channel: &mut T) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.emulate_send(channel)?;
        self.ms_timestamp.emulate_send(channel)
    }
    fn send<T: crate::CommChannel>(
        &self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.send(channel)?;
        self.ms_timestamp.send(channel)
    }

    fn recv<T: crate::CommChannel>(
        &mut self,
        channel: &mut T,
    ) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.recv(channel)?;
        self.ms_timestamp.recv(channel)
    }

    fn try_recv<T: crate::CommChannel>(&mut self, channel: &mut T) -> Result<(), crate::CommChannelError> {
        self.sec_timestamp.try_recv(channel)?;
        self.ms_timestamp.recv(channel)
    }
}
