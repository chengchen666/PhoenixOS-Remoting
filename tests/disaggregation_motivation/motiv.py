# 2-D Vector Bin Packing problem

# setting of our A100 machine
CPU, GPU = 112, 8

# model A: Non-disaggregation
# Greedy algo: First-fit-decreasing
def ND(items):
    # step 1: Ordering the input list by descending size, first sort in GPU
    items.sort(key=lambda x: x[1], reverse=True)

    # step 2: First-fit
    bins = []
    for item in items:
        for bin in bins:
            if bin[0] >= item[0] and bin[1] >= item[1]:
                bin[0] -= item[0]
                bin[1] -= item[1]
                break
        else:
            bins.append([CPU - item[0], GPU - item[1]])

    return len(bins)

# model B: Disaggregation
# Just need to evaluate max(cpu_need, gpu_need)
def D(items):
    CPU_need = sum([item[0] for item in items])
    GPU_need = sum([item[1] for item in items])
    
    # upper round to integer
    return max((CPU_need + CPU - 1) // CPU, (GPU_need + GPU - 1) // GPU)


# report the utilization numbers
def report(items, machines):
    CPU_need = sum([item[0] for item in items])
    GPU_need = sum([item[1] for item in items])
    CPU_total = machines * CPU
    GPU_total = machines * GPU
    print("Number of machines:", machines)
    print(f"CPU utilization: {CPU_need} / {CPU_total} = {CPU_need / CPU_total}")
    print(f"GPU utilization: {GPU_need} / {GPU_total} = {GPU_need / GPU_total}")


# input: items, a list of 2-d vectors, value in [0, 1]^2
items = [[8, 1]] * 269 +  \
        [[12, 1]] * 219 + \
        [[24, 2]] * 27 +  \
        [[32, 4]] * 61 +  \
        [[48, 4]] * 55 +  \
        [[64, 8]] * 45 +  \
        [[82, 8]] * 25 +  \
        [[96, 8]] * 299
        
print(len(items))
assert len(items) == 1000

print("Cluster 1")
print("Non-disaggregation:")
report(items, ND(items))
print("Disaggregation:")
report(items, D(items))


print("")
print("")
print("")


items = [[4, 1]] * 71 +  \
        [[4, 4]] * 119 + \
        [[4, 10]] * 71 +  \
        [[8, 1]] * 147 +  \
        [[16, 1]] * 222 +  \
        [[24, 1]] * 209 +  \
        [[40, 1]] * 52 +  \
        [[48, 2]] * 76 +  \
        [[96, 4]] * 33
        
print(len(items))
assert len(items) == 1000

print("Cluster 2")
print("Non-disaggregation:")
report(items, ND(items))
print("Disaggregation:")
report(items, D(items))
