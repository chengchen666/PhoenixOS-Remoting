import os
import subprocess
import time
import json
import signal

PROJ_ROOT = '/workspace/xpuremoting'
T = 10
comm = 'shm'

KEY_LIST = ['ser', 'send', 'recv', 'raw']
env = os.environ.copy()
env['RUST_LOG'] = 'error'
f = lambda x: f'{PROJ_ROOT}/{x}'
env['NETWORK_CONFIG'] = f('config.toml') if comm == 'shm' else f('config-rdma.toml')

def get_res():
    handler = subprocess.run(f'python3 handle.py', env=env, shell=True, capture_output=True, text=True)
    assert(handler.returncode == 0)
    res = handler.stdout
    res = json.loads(res.replace("'", '"'))
    return res


def get_avg(lis):
    s = {k: 0 for k in lis[0]}
    for r in lis:
        for k, v in r.items():
            s[k] += v
    avg = {k: v / len(lis) for k, v in s.items()}
    return avg


def run(client_cmd, cwd='./tests/cuda_api', compile_cmd=''):
    if compile_cmd:
        subprocess.run(compile_cmd.split(), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    s = subprocess.Popen('./target/release/server', env=env)
    print(s)
    time.sleep(1)
    c = subprocess.run(client_cmd.split(), env=env, cwd=cwd)
    print(c)
    assert(c.returncode == 0)

    s.terminate()
    s.wait()
    time.sleep(1)
    return get_res()


def run_api(cmd, timer_feature='timer'):
    if not timer_feature:
        timer_feature = 'timer'
    
    print('========== without opt ==========')
    print(f'cargo build --features {timer_feature},{comm} --release')
    without_opt = []
    for i in range(T):
        res = run(
            cmd,
            compile_cmd=(f'cargo build --features {timer_feature},{comm} --release' if i == 0 else '')
        )
        without_opt.append(res)
    without_opt = get_avg(without_opt)

    print('========== with opt ==========')
    print(f'cargo build --features {timer_feature},async_api,local,shadow_desc,{comm} --release')
    with_opt = []
    for i in range(T):
        res = run(
            cmd,
            compile_cmd=(f'cargo build --features {timer_feature},async_api,local,shadow_desc,{comm} --release' if i == 0 else '')
        )
        with_opt.append(res)
    with_opt = get_avg(with_opt)

    return [without_opt[k] for k in KEY_LIST] + [with_opt['total']]


def run_apis():
    mp = {
        # 'cudaStreamSynchronize':        ['test_stream_synchronize', ''],
        # 'cudnnConvolutionForward':      ['test_convolution_forward', 'timer_conv'],
        'cudaLaunchKernel':             ['test_launch_kernel', 'timer_kernel'],
        # 'cudnnCreateTensorDescriptor':  ['test_create_tensor_descriptor', ''],
        # 'cudaGetDevice':                ['test_device', ''],
    }
    results = {}
    for k, v in mp.items():
        res = run_api(f'./startclient.sh ./build/{v[0]}', timer_feature=v[1])
        results[k] = res
    
    for k, v in results.items():
        print(f'{k}: {v}')
    print(results)


def run_memcpy(ty):
    assert(ty in ['h2d', 'd2h'])
    rng = [2 ** i for i in range(15, 25)]
    rng.reverse()
    dic = {}
    for sz in rng:
        env['MEMORY_SIZE'] = str(sz)
        res = run_api(
            f'./startclient.sh ./build/test_memcpy_{ty}',
            timer_feature='timer_memcpy'
        )
        dic[sz] = res
    for k, v in dic.items():
        print(f'{k}: {v}')
    print(dic)


if __name__ == '__main__':
    run_apis()

    # run_memcpy('h2d')
    # run_memcpy('d2h')
