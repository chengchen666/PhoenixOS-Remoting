use super::*;
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::ptr;

pub const SHM_BUFFER_SIZE: usize = 104857520;
pub const SHM_NAME_STOC: &str = "/stoc";
pub const SHM_NAME_CTOS: &str = "/ctos";

// TODO: split the abstraction of SharedMemory and RingBuffer
pub struct SharedMemoryBuffer {
    shm_name: String,
    shm_len: usize,
    shm_ptr: *mut libc::c_void,
    buf_size: usize,
    buf: *mut u8,
    buf_head: *mut usize,
    buf_tail: *mut usize,
}

impl SharedMemoryBuffer {
    pub fn new(
        privilege: BufferPrivilege,
        shm_name: &str,
        buf_size: usize,
    ) -> Result<Self, DeviceBufferError> {
        let shm_name = String::from(shm_name);

        match privilege {
            BufferPrivilege::BufferHost => Self::host_init(shm_name, buf_size),
            BufferPrivilege::BufferGuest => Self::guest_init(shm_name, buf_size),
        }
    }

    fn host_init(shm_name: String, buf_size: usize) -> Result<Self, DeviceBufferError> {
        let shm_name_ = CString::new(shm_name.clone()).unwrap();

        // open a shared memory
        let fd: RawFd = unsafe {
            libc::shm_open(
                shm_name_.as_ptr(),
                libc::O_CREAT | libc::O_TRUNC | libc::O_RDWR,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        if fd == -1 {
            error!("Error on shm_open");
            return Err(DeviceBufferError::InvalidOperation);
        }

        // two extra usize for head and tail
        let shm_len = buf_size + std::mem::size_of::<usize>() * 2;
        if unsafe { libc::ftruncate(fd, shm_len as libc::off_t) } == -1 {
            error!("Error on ftruncate");
            unsafe { libc::shm_unlink(shm_name_.as_ptr()) };
            return Err(DeviceBufferError::InvalidOperation);
        }

        // map the shared memory to the process's address space
        let shm_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                shm_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if shm_ptr == libc::MAP_FAILED {
            error!("Error on mmap");
            unsafe { libc::shm_unlink(shm_name_.as_ptr()) };
            return Err(DeviceBufferError::InvalidOperation);
        }

        // initialize the buffer variables
        let buf = shm_ptr as *mut u8;
        let buf_head = unsafe { buf.offset(buf_size as isize) as *mut usize };
        let buf_tail = unsafe { buf_head.offset(1) };
        // set the head and tail to 0
        unsafe {
            *buf_head = 0;
            *buf_tail = 0;
        }

        Ok(Self {
            shm_name,
            shm_len,
            shm_ptr,
            buf_size,
            buf,
            buf_head,
            buf_tail,
        })
    }

    fn guest_init(shm_name: String, buf_size: usize) -> Result<Self, DeviceBufferError> {
        let shm_name_ = CString::new(shm_name.clone()).unwrap();

        // open a shared memory
        let fd: RawFd = unsafe {
            libc::shm_open(
                shm_name_.as_ptr(),
                libc::O_RDWR,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        if fd == -1 {
            error!("Error on shm_open");
            return Err(DeviceBufferError::InvalidOperation);
        }

        // two extra usize for head and tail
        let shm_len = buf_size + std::mem::size_of::<usize>() * 2;

        // map the shared memory to the process's address space
        let shm_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                shm_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if shm_ptr == libc::MAP_FAILED {
            error!("Error on mmap");
            unsafe { libc::shm_unlink(shm_name_.as_ptr()) };
            return Err(DeviceBufferError::InvalidOperation);
        }

        // initialize the buffer variables
        let buf = shm_ptr as *mut u8;
        let buf_head = unsafe { buf.offset(buf_size as isize) as *mut usize };
        let buf_tail = unsafe { buf_head.offset(1) };

        Ok(Self {
            shm_name,
            shm_len,
            shm_ptr,
            buf_size,
            buf,
            buf_head,
            buf_tail,
        })
    }
}

impl Drop for SharedMemoryBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.shm_ptr, self.shm_len);
            let shm_name_ = CString::new(self.shm_name.clone()).unwrap();
            libc::shm_unlink(shm_name_.as_ptr());
        }
    }
}

impl SharedMemoryBuffer {
    // helper method calculating the capacity to write once
    fn write_capacity(&self, read_head: usize) -> usize {
        let read_head = if read_head == 0 {
            self.buf_size
        } else {
            read_head
        };
        unsafe {
            if *self.buf_tail >= read_head {
                self.buf_size - *self.buf_tail
            } else {
                read_head - *self.buf_tail - 1
            }
        }
    }

    // helper method calculating the capacity to read once
    fn read_capacity(&self, read_tail: usize) -> usize {
        if read_tail >= unsafe { *self.buf_head } {
            read_tail - unsafe { *self.buf_head }
        } else {
            self.buf_size - unsafe { *self.buf_head }
        }
    }
}

impl DeviceBuffer for SharedMemoryBuffer {
    // Method to put bytes into the shared buffer
    fn put_bytes(
        &self,
        src: &[u8],
        mode: Option<IssuingMode>,
    ) -> Result<usize, DeviceBufferError> {
        let mut len = src.len();
        let mut offset = 0;
        let mode = mode.unwrap_or(IssuingMode::SyncIssuing);

        while len > 0 {
            // buf_head can be modified by the other side at any time
            // so we need to read it at the beginning and assume it is not changed
            let read_head = unsafe { *self.buf_head };
            if (unsafe { *self.buf_tail } + 1) % self.buf_size == read_head {
                self.flush_out(Some(mode))?;
            }

            let current: usize = std::cmp::min(self.write_capacity(read_head), len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(offset),
                    self.buf.add(*self.buf_tail),
                    current,
                );
            }
            unsafe { *self.buf_tail = (*self.buf_tail + current) % self.buf_size };
            offset += current;
            len -= current;
        }

        Ok(offset)
    }

    fn flush_out(&self, mode: Option<IssuingMode>) -> Result<(), DeviceBufferError> {
        let _mode = mode.unwrap_or(IssuingMode::SyncIssuing);
        while unsafe { (*self.buf_tail + 1) % self.buf_size == *self.buf_head } {
            // Busy-waiting
        }
        Ok(())
    }

    // Method to get bytes from the shared buffer
    fn get_bytes(
        &self,
        dst: &mut [u8],
        mode: Option<IssuingMode>,
    ) -> Result<usize, DeviceBufferError> {
        let mut len = dst.len();
        let mut offset = 0;
        let mode = mode.unwrap_or(IssuingMode::SyncIssuing);

        while len > 0 {
            // buf_tail can be modified by the other side at any time
            // so we need to read it at the beginning and assume it is not changed
            let read_tail = unsafe { *self.buf_tail };
            if unsafe { *self.buf_head } == read_tail {
                self.fill_in(Some(mode))?;
            }

            let current = std::cmp::min(self.read_capacity(read_tail), len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf.add(*self.buf_head),
                    dst.as_mut_ptr().add(offset),
                    current,
                );
            }
            unsafe { *self.buf_head = (*self.buf_head + current) % self.buf_size };
            offset += current;
            len -= current;
        }

        Ok(offset)
    }

    // Method to fill in the buffer
    fn fill_in(&self, mode: Option<IssuingMode>) -> Result<(), DeviceBufferError> {
        let _mode = mode.unwrap_or(IssuingMode::SyncIssuing);
        while unsafe { *self.buf_head } == unsafe { *self.buf_tail } {
            // Busy-waiting
        }
        Ok(())
    }
}

unsafe impl Sync for SharedMemoryBuffer {}
unsafe impl Send for SharedMemoryBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity() {
        let buf = SharedMemoryBuffer::new(
            BufferPrivilege::BufferHost,
            "/test",
            1024 * 1024,
        ).unwrap();

        assert_eq!(buf.write_capacity(0), buf.buf_size - 1);
        // set tail to buf_size
        unsafe {
            *buf.buf_tail = buf.buf_size;
        }
        assert_eq!(buf.write_capacity(0), 0);

        assert_eq!(buf.read_capacity(0), 0);
        // set head to buf_size/2
        unsafe {
            *buf.buf_head = buf.buf_size / 2;
        }
        assert_eq!(buf.read_capacity(0), buf.buf_size / 2);
    }
}
