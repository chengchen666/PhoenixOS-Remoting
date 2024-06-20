import os
import subprocess
import time

PROJ_ROOT = '/workspace/xpuremoting'
ITER = 20

env = os.environ.copy()
env['RUST_LOG'] = 'error'

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


def run(app, model_path, batch, cwd='./tests/apps', comm='shm', opt=False):
    compile_cmd = f"cargo build --features {comm}{',async_api,shadow_desc,local' if opt else ''} --release"
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


def run_app(app, model_path, batch=1):
    res = []

    batch = str(batch)
    res.append(run_local(app, model_path, batch))

    res.append(run(app, model_path, batch, comm='shm', opt=False))
    res.append(run(app, model_path, batch, comm='shm', opt=True))

    res.append(run(app, model_path, batch, comm='rdma', opt=False))
    res.append(run(app, model_path, batch, comm='rdma', opt=True))

    return res


def run_apps():
    get_path = lambda x: f'{PROJ_ROOT}/tests/apps/{x}'
    mp = {
        # 'infer/resnet/inference.py': ('', [1, 64]),
        # 'infer/sd/inference.py': (get_path('infer/sd/stable-diffusion-v1-4'), [1]),
        # 'infer/bert/inference.py': (get_path('infer/bert/bert-base-uncased'), [1, 64]),
        # 'infer/gpt2/inference.py': (get_path('infer/gpt2/gpt2'), [4, 512]),

        'train/resnet/train.py': ('', [64]),
        'train/sd/train.py': ('', [1]),
        'train/bert/train.py': ('', [64]),
    }

    results = {}
    for app, v in mp.items():
        for batch in v[1]:
            res = run_app(app, v[0], batch)
            results[(app, batch)] = res
    
    for k, v in results.items():
        print(f'{k}: {v}')
    print(results)


if __name__ == '__main__':
    run_apps()
