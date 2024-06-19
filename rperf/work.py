
from api import API
from baseline_model.model import model as baseline_model
from disaggregation_model.model import model as disaggregation_model
from sql_parser.parser import parse as parse_sql
import sys


apis = []
rperf_data_dir = sys.argv[1]
gpu_time_sql = f"{rperf_data_dir}/gpu_time.sqlite"
kernel_group_sql = f"{rperf_data_dir}/kernel_group.sqlite"
vanilla_log = f"{rperf_data_dir}/vanilla_rperf.log"

kernels_block, memcpys_block = parse_sql(gpu_time_sql, kernel_group_sql)

with open(vanilla_log, "r") as file:
    lines = file.readlines()
    for line in lines:
        name, Process, Gap = line.strip().split(",")
        api = API(name)
        api.set_Network(0, 0)
        api.set_Process(float(Process))
        api.set_Gap(float(Gap))
        if name == "cudaLaunchKernel" or \
            name == "cublasSgemm_v2" or name == "cublasSgemmStridedBatched" or \
            name == "cudnnConvolutionForward" or name == "cudnnBatchNormalizationForwardInference":
            api.set_Block(kernels_block.pop(0))
        elif name == "cudaMemcpyAsyncDeviceToDevice" or name == "cudaMemcpyAsyncDeviceToHost" or name == "cudaMemcpyAsyncHostToDevice":
            api.set_Block(memcpys_block.pop(0))
        elif name == "cudaStreamSynchronize":
            api.set_Process(2) # own Process time should be small
        apis.append(api)

assert len(kernels_block) == 0
assert len(memcpys_block) == 0

# # fix cudaStreamSynchonize exe overhead (itself actually has no overhead, just wait for previous blocking API)
# prev_api = API("dummy")
# for api in apis:
#     if api.name == "cudaStreamSynchronize":
#         prev_api.Exe += api.Exe
#         api.set_Exe(0)
#     prev_api = api



#### Baseline model

prev_api = API("dummy")
for api in apis:
    baseline_model(prev_api, api)
    prev_api = api

# complete_{n}
print(f"Baseline end time: {apis[-1].complete_time} us")

baseline_time = apis[-1].complete_time



#### Disaggregation model

def calc(apis, baseline_time, RTT):
    for api in apis:
        api.set_Network(RTT / 2, RTT / 2)

    # just for random thought
    # for api in apis:
    #     if api.name == "cudaGetDevice":
    #         api.Gap = min(api.Gap, 20)

    prev_api = API("dummy")
    for api in apis:
        disaggregation_model(prev_api, api)
        prev_api = api

    disaggregation_time = apis[-1].complete_time

    print(f"Disaggregation, RTT = {RTT} us")
    print(f"           end time = {apis[-1].complete_time} us")
    print(f"           Overhead = {disaggregation_time - baseline_time} us")


rtts = [i for i in range(0, 41, 5)]
for rtt in rtts:
    calc(apis, baseline_time, rtt)

calc(apis, baseline_time, 3.4)
