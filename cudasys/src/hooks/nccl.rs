use crate::types::nccl::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 3200)]
fn ncclGetVersion(version: *mut c_int) -> ncclResult_t;

#[cuda_hook(proc_id = 3201)]
fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;

#[cuda_hook(proc_id = 3202)]
fn ncclCommInitRank(
    comm: *mut ncclComm_t,
    nranks: c_int,
    commId: ncclUniqueId,
    rank: c_int,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3203)]
fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3204)]
fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3205)]
fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;

#[cuda_hook(proc_id = 3206)]
fn ncclGroupStart() -> ncclResult_t;

#[cuda_hook(proc_id = 3207)]
fn ncclGroupEnd() -> ncclResult_t;

#[cuda_hook(proc_id = 3208)]
fn ncclSend(
    #[device] sendbuff: *const c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

#[cuda_hook(proc_id = 3209)]
fn ncclRecv(
    #[device] recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;
