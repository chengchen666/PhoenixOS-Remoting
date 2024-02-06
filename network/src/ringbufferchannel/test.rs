#[cfg(test)]
mod tests {
    use super::super::{LocalChannelBufferManager, RingBuffer};
    use crate::CommChannel;
    use super::super::channel::META_AREA;

    #[test]
    fn basic_send_receive() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let data_to_send = [1, 2, 3, 4, 5];
        let mut receive_buffer = [0u8; 5];

        buffer.send(&data_to_send).unwrap();
        buffer.recv(&mut receive_buffer).unwrap();

        assert_eq!(receive_buffer, data_to_send);
    }

    #[test]
    fn partial_receive() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let data_to_send = [1, 2, 3, 4, 5];
        let mut receive_buffer = [0u8; 3];

        buffer.send(&data_to_send).unwrap();
        buffer.recv(&mut receive_buffer).unwrap();

        assert_eq!(receive_buffer, [1, 2, 3]);
    }

    #[test]
    fn wrap_around() {
        println!("Wrap around test");
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(5 + META_AREA));

        let first_send = [1, 2, 3];
        let second_send = [4, 5, 6];
        let mut first_recv = [0u8; 3];
        let mut second_recv = [0u8; 3];

        buffer.send(&first_send).unwrap();
        buffer.recv(&mut first_recv).unwrap(); // Create a gap for wrap-around
        buffer.send(&second_send).unwrap();
        buffer.recv(&mut second_recv).unwrap();

        assert_eq!(first_recv, [1, 2, 3]);
        assert_eq!(second_recv, [4, 5, 6]);
    }

    // TBD
}
