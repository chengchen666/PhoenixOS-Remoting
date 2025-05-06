use crate::{CommChannelError, CommChannelInner, CommChannelInnerIO, RawMemory, RawMemoryMut};

use std::io::{Read, Write};
use std::result::Result;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::io::BufWriter;

pub struct TcpChannel {
    stream:  Arc<Mutex<Option<TcpStream>>>,   // for client-side connection
    write_buf:  Arc<Mutex<Option<BufWriter<TcpStream>>>>,   // for client-side connection
    listener: Option<TcpListener>, // for server-side listener
}

impl TcpChannel {
    pub fn new_server(address: &str) -> Result<Self, std::io::Error> {
        let listener = TcpListener::bind(address)?;
        // println!("Server listening on {}", address);
        Ok(TcpChannel {
            stream: Arc::new(Mutex::new(None)),
            write_buf: Arc::new(Mutex::new(None)),
            listener: Some(listener),
        })
    }

    pub fn new_server_with_stream(stream: std::net::TcpStream, is_sender: bool) -> Result<Self, std::io::Error> {
        if is_sender {
            Ok(TcpChannel {
                stream: Arc::new(Mutex::new(None)),
                write_buf: Arc::new(Mutex::new(Some(BufWriter::new(stream)))),
                listener: None,
            })
        } else {
            Ok(TcpChannel {
                stream: Arc::new(Mutex::new(Some(stream))),
                write_buf: Arc::new(Mutex::new(None)),
                listener: None,
            })
        }
    }

    pub fn new_client(address: &str, is_sender: bool) -> Result<Self, std::io::Error> {
        let stream = TcpStream::connect(address)?;
        // println!("Connected to {}", address);
        if is_sender {
            Ok(TcpChannel {
                stream: Arc::new(Mutex::new(None)),
                write_buf: Arc::new(Mutex::new(Some(BufWriter::new(stream)))),
                listener: None,
            })
        } else {
            Ok(TcpChannel {
                stream: Arc::new(Mutex::new(Some(stream))),
                write_buf: Arc::new(Mutex::new(None)),
                listener: None,
            })
        }
    }

    pub fn new_client_with_stream(stream: std::net::TcpStream, is_sender: bool) -> Result<Self, std::io::Error> {
        if is_sender {
            Ok(TcpChannel {
                stream: Arc::new(Mutex::new(None)),
                write_buf: Arc::new(Mutex::new(Some(BufWriter::new(stream)))),
                listener: None,
            })
        } else {
            Ok(TcpChannel {
                stream: Arc::new(Mutex::new(Some(stream))),
                write_buf: Arc::new(Mutex::new(None)),
                listener: None,
            })
        }
    }

    pub fn accept_connection(&mut self) -> Result<(), std::io::Error> {
        if let Some(listener) = &self.listener {
            match listener.accept() {
                Ok((stream, _)) => {
                    self.stream = Arc::new(Mutex::new(Some(stream)));
                    Ok(())
                },
                Err(e) => Err(e),
            }
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Not a server channel",
            ))
        }
    }

    // Close the connection (for both server and client)
    fn close_connection(&mut self) -> Result<(), std::io::Error> {
        let mut locked_write_buffer = self.write_buf.lock().unwrap();
        if let Some(wirte_buffer) = locked_write_buffer.as_mut() {
            wirte_buffer.flush()?;
            let inner_stream = wirte_buffer.get_ref();
            inner_stream.shutdown(std::net::Shutdown::Both)
        } else {
            let locked_stream  = self.stream.lock().unwrap();
            if let Some(ref stream) = *locked_stream {
                stream.shutdown(std::net::Shutdown::Both)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "No active stream to close",
                ))
            }
        }
    }
}

impl Drop for TcpChannel {
    fn drop(&mut self) {
        self.close_connection().unwrap();
    }
}

impl CommChannelInner for TcpChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        let mut locked_write_buffer  = self.write_buf.lock().unwrap();
        if let Some(ref mut wirte_buffer) = *locked_write_buffer {
            wirte_buffer.flush().unwrap();
        }
        Ok(())
    }
}

impl CommChannelInnerIO for TcpChannel {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        if src.len == 0 {
            return Ok(0);
        }
        if let Some(slice) = src.as_slice() {
            let mut locked_write_buffer  = self.write_buf.lock().unwrap();
            if let Some(ref mut wirte_buffer) = *locked_write_buffer {
                match wirte_buffer.write_all(slice) {
                    Ok(()) => {
                        return Ok(slice.len());
                    },
                    Err(_) => {
                        return Err(CommChannelError::IoError);
                    }
                }
            }
        }
        Err(CommChannelError::IoError)
    }

    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
       self.put_bytes(src)
    }

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        if dst.len == 0 {
            return Ok(0);
        }
        if let Some(slice) = dst.as_mut_slice() {
            let mut locked_stream  = self.stream.lock().unwrap();
            if let Some(ref mut stream) =  *locked_stream {
                match stream.read_exact(slice) {
                    Ok(()) => {
                        return Ok(slice.len());
                    },
                    Err(_) => {
                        return Err(CommChannelError::IoError);
                    }
                }
            }
        }
        Err(CommChannelError::IoError)
    }

    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
       self.get_bytes(dst)
    }

    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.try_get_bytes(dst)
    }
}
