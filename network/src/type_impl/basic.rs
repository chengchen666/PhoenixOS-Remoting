use super::*;

impl Transportable for i32 {
    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let memory = RawMemory::new(self, std::mem::size_of::<Self>());
        match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
        let mut memory = RawMemoryMut::new(self, std::mem::size_of::<Self>());
        match channel.get_bytes(&mut memory)? == std::mem::size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }
}

impl Transportable for usize {
    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let memory = RawMemory::new(self, std::mem::size_of::<Self>());
        match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
        let mut memory = RawMemoryMut::new(self, std::mem::size_of::<Self>());
        match channel.get_bytes(&mut memory)? == std::mem::size_of::<Self>() {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }
}

impl<S> Transportable for [S] {
    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let len = self.len() * std::mem::size_of::<S>();
        let memory = RawMemory::from_ptr(self.as_ptr() as *const u8, len);
        match channel.put_bytes(&memory)? == len {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
        let len = self.len() * std::mem::size_of::<S>();
        let mut memory = RawMemoryMut::from_ptr(self.as_mut_ptr() as *mut u8, len);
        match channel.get_bytes(&mut memory)? == len {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }
}

impl<S> Transportable for Vec<S>
where
    S: Default + Clone,
{
    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let len: usize = self.len();
        len.send(channel)?;
        self.as_slice().send(channel)
    }

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
        let mut len: usize = 0;
        len.recv(channel)?;
        self.resize(len, S::default());
        self.as_mut_slice().recv(channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ringbufferchannel::{channel::META_AREA, LocalChannelBufferManager, RingBuffer};

    /// Test i32 Transportable impl
    #[test]
    fn test_i32_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let a = 123;
        let mut b = 0;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    /// Test [u8] Transportable impl
    #[test]
    fn test_u8_array_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let a = [1u8, 2, 3, 4, 5];
        let mut b = [0u8; 5];
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    /// Test [i32] Transportable impl
    #[test]
    fn test_i32_array_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(50 + META_AREA));
        let a = [1i32, 2, 3, 4, 5];
        let mut b = [0i32; 5];
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    /// Test Vec<i32> Transportable impl
    #[test]
    fn test_vec_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(50 + META_AREA));
        let a = vec![1, 2, 3, 4, 5];
        let mut b = vec![0; 5];
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }
}
