use super::*;

impl SerializeAndDeserialize for i32 {
    fn to_bytes(&self) -> Result<Vec<u8>, CommChannelError> {
        let buf = self.to_le_bytes();
        Ok(buf.to_vec())
    }

    fn from_bytes(&mut self, src: &Vec<u8>) -> Result<(), CommChannelError> {
        if src.len() < std::mem::size_of::<i32>() {
            return Err(CommChannelError::IoError);
        }
        *self = i32::from_le_bytes(src[0..std::mem::size_of::<i32>()].try_into().unwrap());
        Ok(())
    }
}
