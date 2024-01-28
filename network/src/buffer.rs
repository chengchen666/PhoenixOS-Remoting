use std::error::Error;
use std::{fmt, mem, ptr, slice};

/// A custom error type for buffer operations.
#[derive(Debug)]
pub enum BufferError {
    InsufficientSpace,
    InvalidOperation,
}

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buffer Error: {:?}", self)
    }
}

impl Error for BufferError {}

/// A raw buffer implementation for high-performance use cases.
pub struct RawBuffer {
    buffer: *mut u8,
    capacity: usize,
    size: usize,
    off: usize,
}

impl RawBuffer {
    /// Creates a new buffer with the specified capacity.
    pub fn new(capacity: usize) -> RawBuffer {
        let buffer = unsafe { libc::malloc(capacity) as *mut u8 };
        RawBuffer {
            buffer,
            capacity,
            size: 0,
            off: 0,
        }
    }

    /// Appends data to the buffer. Returns an error if the buffer is full.
    pub unsafe fn append<T>(&mut self, item: &T) -> Result<(), BufferError> {
        // XD: seems rust will align the object sizes
        let item_size = mem::size_of::<T>();
        if self.size + item_size > self.capacity {
            return Err(BufferError::InsufficientSpace);
        }
        ptr::copy_nonoverlapping(item as *const T, self.buffer.add(self.size) as *mut T, 1);
        self.size += item_size;

        Ok(())
    }

    /// Extracts a value of type `T` from the buffer at the current offset.
    /// Unsafe because it assumes that there is a valid `T` at that offset and that
    /// the memory is aligned correctly for `T`.
    pub unsafe fn extract<T>(&mut self) -> Result<T, BufferError>
    where
        T: Copy, // Ensure that T can be safely copied
    {
        // Check that the offset is within the bounds of the buffer and that there is enough space to read a T
        if self.off + mem::size_of::<T>() > self.size {
            return Err(BufferError::InsufficientSpace);
        }

        // Ensure the memory at the given offset is properly aligned for type T
        if (self.buffer.add(self.off) as usize) % mem::align_of::<T>() != 0 {
            return Err(BufferError::InvalidOperation);
        }

        // Read the value of type T from the buffer
        let value = ptr::read_unaligned(self.buffer.add(self.off) as *const T);
        self.off += mem::size_of::<T>();
        Ok(value)
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.buffer, self.size) }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.buffer as *mut libc::c_void);
        }
    }
}

impl fmt::Debug for RawBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let contents = unsafe { std::slice::from_raw_parts(self.buffer, self.size) };
        f.debug_struct("RawBuffer")
            .field("capacity", &self.capacity)
            .field("size", &self.size)
            .field("contents", &format_args!("{:?}", contents))
            .field("off", &self.off)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::RawBuffer;
    use std::mem::size_of;

    #[test]
    fn test_buffer_creation() {
        let buffer = RawBuffer::new(32);
        assert_eq!(buffer.capacity, 32);
        assert_eq!(buffer.size, 0);
    }

    #[test]
    fn test_append_and_retrieve() {
        let mut buffer = RawBuffer::new(64);
        let value: u32 = 42;

        unsafe {
            assert!(buffer.append(value).is_ok());
        }

        let buffer_slice = buffer.as_slice();
        let retrieved_value =
            u32::from_ne_bytes(buffer_slice[0..size_of::<u32>()].try_into().unwrap());
        assert_eq!(retrieved_value, value);

        let val = unsafe { buffer.extract::<u32>() }.unwrap();
        assert_eq!(val, value);
    }

    #[test]
    fn test_insufficient_space_error() {
        let mut buffer = RawBuffer::new(4); // small buffer
        let value: u64 = 123456789; // 8 bytes, too large for the buffer

        let result = unsafe { buffer.append(value) };
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_capacity_limit() {
        let mut buffer = RawBuffer::new(8); // buffer for exactly two u32 values
        unsafe {
            assert!(buffer.append(1u32).is_ok());
            assert!(buffer.append(2u32).is_ok());
        }

        assert_eq!(buffer.size, 8);
        assert_eq!(buffer.capacity, 8);

        let result = unsafe { buffer.append(3u32) }; // should fail, no space left
        assert!(result.is_err());
    }
}
