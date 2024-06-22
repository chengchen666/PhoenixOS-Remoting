import os
import subprocess
import time

PROJ_ROOT = '/workspace/xpuremoting'
ITER = 20

env = os.environ.copy()
env['RUST_LOG'] = 'error'

def get_opt(level):
    opt = ''
    if level >= 1:
        opt += ',async_api'
    if level >= 2:
        opt += ',shadow_desc'
    if level >= 3:
        opt += ',local'
    return opt


def get_time(client):
    output = client.stdout.splitlines()
    return (float(output[-1]) * 1000) / int(ITER)


def run_local(app, model_path, batch, cwd='./tests/apps'):
    cmd = ['python3', app, str(ITER), batch]
    if model_path:
        cmd += [model_path]
    print(cmd)
    c = subprocess.run(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE)
    assert(c.returncode == 0)

    return get_time(c)


def run(app, model_path, batch, cwd='./tests/apps', comm='shm', opt=0):
    compile_cmd = f"cargo build --features {comm}{get_opt(opt)} --release"
    print(compile_cmd)
    subprocess.run(compile_cmd.split(), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    f = lambda x: f'{PROJ_ROOT}/{x}'
    env['NETWORK_CONFIG'] = f('config.toml') if comm == 'shm' else f('config-rdma.toml')
    print(env['NETWORK_CONFIG'])

    s = subprocess.Popen('./target/release/server', env=env)
    print(s)
    time.sleep(1)

    cmd = ['./run.sh', app, str(ITER), batch]
    if model_path:
        cmd += [model_path]
    print(cmd)
    c = subprocess.run(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE)
    assert(c.returncode == 0)

    s.terminate()
    s.wait()
    time.sleep(1)
    return get_time(c)


def run_app(app, model_path, batch):
    res = []

    res.append(run_local(app, model_path, batch))

    res.append(run(app, model_path, batch, comm='shm', opt=0))
    res.append(run(app, model_path, batch, comm='shm', opt=3))

    res.append(run(app, model_path, batch, comm='rdma', opt=0))
    res.append(run(app, model_path, batch, comm='rdma', opt=3))

    return res


def run_factor(app, model_path, batch):
    res = []

    # res.append(run_local(app, model_path, batch))
    res.append(run(app, model_path, batch, comm='rdma', opt=3))
    print(res)
    res.append(run(app, model_path, batch, comm='rdma', opt=2))
    print(res)
    res.append(run(app, model_path, batch, comm='rdma', opt=1))
    print(res)
    res.append(run(app, model_path, batch, comm='rdma', opt=0))

    return res


def run_apps():
    get_path = lambda x: f'{PROJ_ROOT}/tests/apps/{x}'
    mp = {
        # 'infer/resnet/inference.py': ('', [1, 64]),
        # 'infer/STABLEDIFFUSION-v1-4/inference.py': (get_path('infer/STABLEDIFFUSION-v1-4/stable-diffusion-v1-4'), [1]),
        # 'infer/BERT-base-uncased/inference.py': (get_path('infer/BERT-base-uncased/bert-base-uncased'), [1, 64]),
        # 'infer/gpt2/inference.py': (get_path('infer/gpt2/gpt2'), [4, 512]),

        # 'train/resnet/train.py': ('', [64]),
        'train/STABLEDIFFUSION-v1-4/train.py': ('', [1]),
        # 'train/BERT-base-uncased/train.py': ('', [64]),
    }

    results = {}
    for app, v in mp.items():
        for batch in v[1]:
            res = run_app(app, v[0], str(batch))
            results[(app, batch)] = res
            print(results)
    
    for k, v in results.items():
        print(f'{k}: {v}')
    print(results)


def run_factors():
    get_path = lambda x: f'{PROJ_ROOT}/tests/apps/{x}'
    mp = {
        # 'infer/resnet/inference.py': ('', [64]),
        # 'infer/STABLEDIFFUSION-v1-4/inference.py': (get_path('infer/STABLEDIFFUSION-v1-4/stable-diffusion-v1-4'), [1]),
        # 'infer/BERT-base-uncased/inference.py': (get_path('infer/BERT-base-uncased/bert-base-uncased'), [64]),
        # 'infer/gpt2/inference.py': (get_path('infer/gpt2/gpt2'), [512]),

        # 'train/resnet/train.py': ('', [64]),
        'train/STABLEDIFFUSION-v1-4/train.py': ('', [1]),
        # 'train/BERT-base-uncased/train.py': ('', [64]),
    }

    results = {}
    for app, v in mp.items():
        for batch in v[1]:
            res = run_factor(app, v[0], str(batch))
            results[(app, batch)] = res
    
    for k, v in results.items():
        print(f'{k}: {v}')
    print(results)


if __name__ == '__main__':
    # run_apps()
    run_factors()
