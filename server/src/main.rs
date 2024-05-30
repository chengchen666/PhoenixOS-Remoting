extern crate log;
extern crate network;
extern crate server;
use log::info;
use network::CONFIG;
use server::*;
use std::{
    env,
    sync::{Arc, Mutex},
    thread,
};

fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "debug");
    }
    if std::env::var("EMU_TYPE").is_err() {
        std::env::set_var("EMU_TYPE", "server");
    }
    env_logger::init();
    launch_server();
}
