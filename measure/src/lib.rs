use std::arch::asm;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{self, Write};

#[macro_use]
extern crate lazy_static;

pub const MEASURE_TOTAL: usize = 0;
pub const MEASURE_CSER: usize = 1;
pub const MEASURE_CSEND: usize = 2;
pub const MEASURE_SRECV: usize = 3;
pub const MEASURE_RAW: usize = 4;
pub const MEASURE_SSER: usize = 5;
pub const MEASURE_SSEND: usize = 6;
pub const MEASURE_CRECV: usize = 7;
const MEASURE_TYPE_LEN: usize = 8;

const CLOCK_FREQUENCY: f64 = 2.2;
const ITER_NUM: usize = 10010;

pub struct Timer {
    start_time: [[u64; MEASURE_TYPE_LEN]; ITER_NUM],
    stop_time: [[u64; MEASURE_TYPE_LEN]; ITER_NUM],
    cnt: usize,
    output_file: String,
}

#[inline]
fn rdtscp() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!("rdtscp", out("rax") lo, out("rdx") hi, options(nomem, nostack));
    }
    ((hi as u64) << 32) | (lo as u64)
}

#[inline]
fn clock2ns(clock: u64) -> f64 {
    clock as f64 / CLOCK_FREQUENCY
}

impl Timer {
    pub fn new(file_name: String) -> Self {
        Timer {
            start_time: [[0; MEASURE_TYPE_LEN]; ITER_NUM],
            stop_time: [[0; MEASURE_TYPE_LEN]; ITER_NUM],
            cnt: 0,
            output_file: file_name,
        }
    }

    #[inline]
    pub fn start(&mut self, ty: usize) {
        self.start_time[self.cnt][ty] = rdtscp();
    }

    #[inline]
    pub fn stop(&mut self, ty: usize) {
        self.stop_time[self.cnt][ty] = rdtscp();
    }

    #[inline]
    pub fn plus_cnt(&mut self) {
        self.cnt += 1;
        if self.cnt == ITER_NUM {
            let _ = self.write();
        }
    }

    pub fn write(&self) -> io::Result<()> {
        // let mut file = OpenOptions::new()
        //     .append(true)
        //     .create(true)
        //     .open(file_name)?;
        let mut file = File::create(self.output_file.clone())?;
        for i in 0..ITER_NUM {
            let mut row_result = Vec::new();
            for j in 0..MEASURE_TYPE_LEN {
                row_result.push(self.start_time[i][j].to_string());
                row_result.push(self.stop_time[i][j].to_string());
            }
            writeln!(file, "{}", &row_result.join(", "))?;
        }
        // writeln!(file, "{}, {}, {}, {}, {}",
        //     self.get_time(MEASURE_TOTAL),
        //     self.get_time(MEASURE_RAW),
        //     self.get_time(MEASURE_SER),
        //     self.get_time(MEASURE_SEND),
        //     self.get_time(MEASURE_WAIT)
        // )?;
        Ok(())
    }

    // #[inline]
    // pub fn get_clock(&self, ty: usize) -> u64 {
    //     self.stop_time[ty] - self.start_time[ty]
    // }

    // #[inline]
    // pub fn get_time(&self, ty: usize) -> f64 {
    //     clock2ns(self.get_clock(ty))
    // }
}

use std::sync::Mutex;
lazy_static! {
    pub static ref CTIMER: Mutex<Timer> =
        Mutex::new(Timer::new("/workspace/xpuremoting/client.out".to_string()));
    pub static ref STIMER: Mutex<Timer> =
        Mutex::new(Timer::new("/workspace/xpuremoting/server.out".to_string()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_test() {
        let mut timer = Timer::new();

        timer.start(MEASURE_TOTAL);
        let mut _sum: u64 = 0;
        for i in 0..10000 {
            _sum += i;
        }
        timer.stop(MEASURE_TOTAL);
        assert!(timer.get_time(MEASURE_TOTAL) > 0.0);
    }
}
