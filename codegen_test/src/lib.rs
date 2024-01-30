extern crate codegen;
extern crate network;

#[cfg(test)]
mod tests {
    use codegen::{gen_deserialize, gen_serialize};
    use network::*;

    gen_serialize!("basic_s", "usize");
    gen_deserialize!("basic_des", "usize");

    #[test]
    fn basic_serialization() {
        let mut buffer = RawBuffer::new(64);
        basic_s(&mut buffer, &12).unwrap();

        let mut val: usize = 0;
        basic_des(&mut buffer, &mut val).unwrap();
        
        assert_eq!(val, 12);
    }
}
