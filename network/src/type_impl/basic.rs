use super::*;

impl SerializeAndDeserialize for i32 {
    fn to_bytes(&self) -> Result<[u8; std::mem::size_of::<Self>()], CommChannelError> {
        let buf = self.to_le_bytes();
        Ok(buf)
    }

    fn from_bytes(&mut self, src: &[u8]) -> Result<(), CommChannelError> {
        if src.len() < std::mem::size_of::<Self>() {
            return Err(CommChannelError::IoError);
        }
        *self = i32::from_le_bytes(src[0..std::mem::size_of::<i32>()].try_into().unwrap());
        Ok(())
    }
}
