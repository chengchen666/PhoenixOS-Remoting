# pub const MEASURE_TOTAL: usize = 0;
# pub const MEASURE_CSER: usize = 1;
# pub const MEASURE_CSEND: usize = 2;
# pub const MEASURE_SRECV: usize = 3;
# pub const MEASURE_RAW: usize = 4;
# pub const MEASURE_SSER: usize = 5;
# pub const MEASURE_SSEND: usize = 6;
# pub const MEASURE_CRECV: usize = 7;
mp = {
    'total': 0,
    'cser': 1,
    'csend': 2,
    'srecv': 3,
    'raw': 4,
    'sser': 5,
    'ssend': 6,
    'crecv': 7,
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

def calc(data):
    ret = []
    lis = ['ser', 'send', 'recv', 'raw', 'total']
    total_time = {}
    for col in lis:
        total_time[col] = 0

    for line in data:
        f = lambda col: line[mp[col] * 2 + 1] - line[mp[col] * 2]
        res = {}
        res['ser'] = f('cser') + f('sser')
        res['send'] = f('csend') + f('ssend')
        res['recv'] = f('crecv') + f('srecv')
        res['raw'] = f('raw')
        res['total'] = f('total')
        for col in lis:
            total_time[col] += res[col]
        ret.append(res)
    return ret, total_time

data = get_data()
clock_count, total_time = calc(data)

for k in total_time:
    total_time[k] /= 2.5
    total_time[k] /= 10000

print(total_time)
