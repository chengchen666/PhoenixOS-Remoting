extern crate log;
extern crate network;
extern crate server;
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
    launch_server();
}
