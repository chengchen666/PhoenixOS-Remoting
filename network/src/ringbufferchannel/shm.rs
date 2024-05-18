use super::ChannelBufferManager;
use log::error;

use std::ffi::CString;
use std::io::Result;
use std::os::unix::io::RawFd;

use errno::errno;

/// A shared memory channel buffer manager
pub struct SHMChannelBufferManager {
    shm_name: String,
    shm_ptr: *mut libc::c_void,
    shm_len: usize,
}

impl SHMChannelBufferManager {
    /// Create a new shared memory channel buffer manager for the server
    /// The name server is more consistent with the remoting library
    pub fn new_server(shm_name: &str, shm_len: usize) -> Result<Self> {
        Self::new_inner(
            shm_name,
            shm_len,
            libc::O_CREAT | libc::O_TRUNC | libc::O_RDWR,
            (libc::S_IRUSR | libc::S_IWUSR) as _,
        )
    }

    pub fn new_client(shm_name: &str, shm_len: usize) -> Result<Self> {
        Self::new_inner(
            shm_name,
            shm_len,
            libc::O_RDWR,
            (libc::S_IRUSR | libc::S_IWUSR) as _,
        )
    }
    fn new_inner(shm_name: &str, shm_len: usize, oflag: i32, sflag: i32) -> Result<Self> {
        let shm_name_c_str = CString::new(shm_name).unwrap();
        let fd: RawFd =
            unsafe { libc::shm_open(shm_name_c_str.as_ptr(), oflag, sflag as _) };

        if fd == -1 {
            error!("Error on shm_open for new_host");
            return Err(std::io::Error::from_raw_os_error(errno().0));
        }

        if unsafe { libc::ftruncate(fd, shm_len as libc::off_t) } == -1 {
            error!("Error on ftruncate");
            unsafe { libc::shm_unlink(shm_name.as_ptr() as _) };
            return Err(std::io::Error::from_raw_os_error(errno().0));
        }

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
            error!("Error on mmap the SHM pointer");
            unsafe { libc::shm_unlink(shm_name.as_ptr() as _) };
            return Err(std::io::Error::from_raw_os_error(errno().0));
        }

        Ok(Self {
            shm_name: String::from(shm_name),
            shm_len,
            shm_ptr: shm_ptr,
        })
    }
}

impl Drop for SHMChannelBufferManager {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.shm_ptr, self.shm_len);
            let shm_name_ = CString::new(self.shm_name.clone()).unwrap();
            libc::shm_unlink(shm_name_.as_ptr());
        }
    }
}

impl ChannelBufferManager for SHMChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.shm_ptr as *mut u8, self.shm_len)
    }

    fn read_at(&self, offset: usize, dst: *mut u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(self.shm_ptr.add(offset) as _, dst, count);
        }
        count
    }

    fn write_at(&self, offset: usize, src: *const u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.shm_ptr.add(offset) as _, count);
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shm_channel_buffer_manager() {
        let shm_name = "/stoc";
        let shm_len = 64;
        let manager = SHMChannelBufferManager::new_server(shm_name, shm_len).unwrap();
        assert_eq!(manager.shm_len, shm_len);
    }
}
