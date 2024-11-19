use server::*;
use std::{
    env,
};

fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "debug");
    }
    env_logger::init();
    // core_affinity::set_for_current(0);
    let config = &*network::CONFIG;
    match config.comm_type.as_str() {
        "shm" => {
            let listener = std::net::TcpListener::bind(&config.daemon_socket).unwrap();
            let mut id = 0;
            while let Ok((stream, _)) = listener.accept() {
                std::thread::spawn(move || {
                    launch_server(config, id, Some(stream));
                });
                id += 1;
            }
        }
        #[cfg(feature = "rdma")]
        "rdma" => launch_server(config, 0, None),
        _ => panic!("Unsupported communication type in config"),
    }
}
