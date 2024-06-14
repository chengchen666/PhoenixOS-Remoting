mp = {
    'start': 0,
    'cser': 1,
    'csend': 2,
    'srecv': 3,
    'sdser': 4,
    'raw': 5,
    'sser': 6,
    'ssend': 7,
    'crecv': 7,
    'crecv': 8,
    'cdser': 9,
    'total': 10,
}

def get_data():
    with open('./client.out', 'r') as f:
        client_lines = f.readlines()
    with open('./server.out', 'r') as f:
        server_lines = f.readlines()
    
    data = []
    for i in range(10, len(client_lines)):
        res = list(map(int, client_lines[i].split(',')))
        s = list(map(int, server_lines[i].split(',')))
        for i in range(len(res)):
            if res[i] == 0 and s[i] != 0:
                res[i] = s[i]
        data.append(res)
    return data

def get_time(line, p=False):
    f = lambda col: max(line[mp[col]] - line[mp[col] - 1], 0)
    if p:
        print(f('cser'), f('sdser'), f('sser'), f('cdser'), f('srecv'), f('crecv'))
    raw = f('raw')
    ser = f('cser') + f('sdser') + f('sser') + f('cdser')
    send = f('csend') + f('ssend')
    recv = f('srecv') + f('crecv')
    total = line[mp['total']] - line[mp['start']]
    return {
        'raw': raw,
        'ser': ser,
        'send': send,
        'recv': recv,
        'total': total,
    }

def calc(data):
    ret = []
    lis = ['ser', 'send', 'recv', 'raw', 'total']
    total_time = {}
    for col in lis:
        total_time[col] = 0

    for line in data:
        res = get_time(line)
        for col in lis:
            total_time[col] += res[col]
        ret.append(res)
    return ret, total_time, len(data)

data = get_data()
clock_count, total_time, total_num = calc(data)

f = lambda col: line[mp[col] * 2 + 1] - line[mp[col] * 2]
for line in data:
    print(line)
    res = get_time(line, True)
    print(res)
    break

for k in total_time:
    total_time[k] /= 2.2
    total_time[k] /= total_num
    total_time[k] /= 1000

print(total_time)
