extern crate device_buffer;

use std::{thread, vec};
use device_buffer::*;

// helper host function
fn host(content: &[u8]) {
    // create a shared memory buffer
    let mut shm_buf =
        SharedMemoryBuffer::new(BufferPrivilege::BufferHost, "/shm_buf", 1024).unwrap();

    // read the length:usize
    let mut buf = [0u8; std::mem::size_of::<usize>()];
    if let Ok(len_read) = shm_buf.get_bytes(&mut buf, None) {
        assert_eq!(len_read, std::mem::size_of::<usize>());
        println!("Host: read length: {:?}", &buf[..len_read]);
    }
    let len = usize::from_le_bytes(buf);
    assert_eq!(len, content.len());

    // read the data from the buffer
    let mut buf = vec![0u8; len];
    if let Ok(len_read) = shm_buf.get_bytes(&mut buf, None) {
        assert_eq!(len_read, len);
        println!("Host: read data: {:?}", &buf[..len_read]);
    }
    assert!(buf == content);
}

// helper guest function
fn guest(content: &[u8]) {
    // create a shared memory buffer
    let mut shm_buf =
        SharedMemoryBuffer::new(BufferPrivilege::BufferGuest, "/shm_buf", 1024).unwrap();

    // write the length:usize
    let len = content.len();
    let buf = len.to_le_bytes();
    if let Ok(len_written) = shm_buf.put_bytes(&buf, None) {
        assert_eq!(len_written, std::mem::size_of::<usize>());
        println!("Guest: write length: {:?}", &buf[..len_written]);
    }

    // write the data to the buffer
    let mut buf = vec![0u8; len];
    buf.copy_from_slice(content);
    if let Ok(len_written) = shm_buf.put_bytes(&buf, None) {
        assert_eq!(len_written, len);
        println!("Guest: write data: {:?}", &buf[..len_written]);
    }
}

#[test]
fn hello_world() {
    let host_thread = thread::spawn(move || {
        host(b"hello world");
    });

    // sleep for 1s
    thread::sleep(Duration::from_secs(1));

    let guest_thread = thread::spawn(move || {
        guest(b"hello world");
    });

    host_thread.join().unwrap();
    guest_thread.join().unwrap();
}
