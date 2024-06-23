
from api import API
from baseline_model.model import model as baseline_model
from disaggregation_model.model import model as disaggregation_model
from typedict.remoting import get_remoting_type
from sql_parser.parser import parse as parse_sql
import sys


apis = []
rperf_data_dir = sys.argv[1]
gpu_time_sql = f"{rperf_data_dir}/gpu_time.sqlite"
kernel_group_sql = f"{rperf_data_dir}/kernel_group.sqlite"
vanilla_log = f"{rperf_data_dir}/vanilla_rperf.log"
xpu_remoting_log = f"{rperf_data_dir}/xpu_remoting.log"

kernels_block, memcpys_block = parse_sql(gpu_time_sql, kernel_group_sql)

with open(vanilla_log, "r") as file:
    lines = file.readlines()
    for line in lines:
        name, Process, Gap = line.strip().split(",")
        if name == "cudaMemcpyAsyncDeviceToDevice" or name == "cudaMemcpyAsyncDeviceToHost" or name == "cudaMemcpyAsyncHostToDevice":
            name = "cudaMemcpyAsync"
        api = API(name)
        api.set_Payload(0, 0)
        api.set_Network(0, 0)
        api.set_Process(float(Process))
        api.set_Gap(float(Gap))
        if name == "cudaLaunchKernel" or \
            name == "cublasSgemm_v2" or name == "cublasSgemmStridedBatched" or \
            name == "cudnnConvolutionForward" or name == "cudnnBatchNormalizationForwardInference" or \
            name == "cudnnBatchNormalizationBackwardEx" or name == "cudnnBatchNormalizationForwardTrainingEx" or \
            name == "cudnnConvolutionBackwardData" or name == "cudnnConvolutionBackwardFilter":
            api.set_Block(kernels_block.pop(0))
        elif name == "cudaMemcpyAsync":
            api.set_Block(memcpys_block.pop(0))
        elif name == "cudaStreamSynchronize":
            api.set_Process(2) # own Process time should be small
        apis.append(api)

assert len(kernels_block) == 0
assert len(memcpys_block) == 0

payloads = []
with open(xpu_remoting_log, "r") as file:
    lines = file.readlines()
    for line in lines:
        name, Serialization, Payload = line.strip().split(",")
        if name == "cuDevicePrimaryCtxGetState":
            continue # drop because not catched in the vanilla log
        payloads.append((name, float(Serialization), int(Payload)))

i = 0
while len(payloads) > 0 and i < len(apis):
    name, serialization, payload_forward = payloads[0][0], payloads[0][1], payloads[0][2]
    if name == "cudaMemcpy":
        name = "cudaMemcpyAsync"
    payloads.pop(0)
    payload_backward = get_remoting_type(name)[1]
    while i < len(apis) and apis[i].name != name:
        i += 1
    if i == len(apis):
        break
    apis[i].set_Serialization(serialization, serialization)
    apis[i].set_Payload(payload_forward, payload_backward)

assert len(payloads) == 0


#### Baseline model

prev_api = API("dummy")
for api in apis:
    baseline_model(prev_api, api)
    prev_api = api

# complete_{n}
print(f"Baseline end time: {apis[-1].complete_time} us")

baseline_time = apis[-1].complete_time



#### Disaggregation model

def calc(apis, baseline_time, RTT, BANDWIDTH):
    for api in apis:
        api.calc_Network(RTT, BANDWIDTH)

    # just for random thought
    # for api in apis:
    #     if api.name == "cudaGetDevice":
    #         api.Gap = min(api.Gap, 20)

    prev_api = API("dummy")
    for api in apis:
        disaggregation_model(prev_api, api)
        prev_api = api

    disaggregation_time = apis[-1].complete_time

    print(f"Disaggregation, RTT = {RTT} us, BANDWIDTH = {BANDWIDTH} GBps")
    print(f"           end time = {apis[-1].complete_time} us")
    print(f"           Overhead = {disaggregation_time - baseline_time} us")


rtts = [i for i in range(0, 41, 5)]
for rtt in rtts:
    calc(apis, baseline_time, rtt, 72.17)

calc(apis, baseline_time, 3.4, 72.17)
