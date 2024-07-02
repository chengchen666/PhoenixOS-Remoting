from api import API
from typedict.blocking import get_blocking_type


def model(previous_api: API, current_api: API):
    if previous_api.name == "dummy":
        previous_api.complete_time = 0
        previous_api.queue_time = 0
    
    blocking_type = get_blocking_type(current_api.name)
    
    # issue_{i} = complete_{i-1} + Gap_{i}
    current_api.issue_time = previous_api.complete_time + current_api.Gap

    # start_{i} = issue_{i}
    current_api.start_time = current_api.issue_time

    # end_{i} = start_{i} + Process_{i}, NonBlocking or GPUBlocking
    #         = max(start_{i} + Process_{i}, queue_{i - 1}) + Block_{i}, CPUBlocking
    if blocking_type == "NonBlocking" or blocking_type == "GPUBlocking":
        current_api.end_time = current_api.start_time + current_api.Process
    elif blocking_type == "CPUBlocking":
        current_api.end_time = max(current_api.start_time + current_api.Process, previous_api.queue_time) + current_api.Block

    # queue_{i} = queue_{i-1}, NonBlocking
    #           = max(start_{i} + Process_{i}, queue_{i - 1}) + Block_{i}, GPUBlocking or CPUBlocking
    if blocking_type == "NonBlocking":
        current_api.queue_time = previous_api.queue_time
    elif blocking_type == "GPUBlocking" or blocking_type == "CPUBlocking":
        current_api.queue_time = max(current_api.start_time + current_api.Process, previous_api.queue_time) + current_api.Block

    # complete_{i} = end_{i}
    current_api.complete_time = current_api.end_time
