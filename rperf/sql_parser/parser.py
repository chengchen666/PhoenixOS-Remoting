import sqlite3
import csv

def parse(gpu_time_sql, kernel_wrapper_sql):
    conn = sqlite3.connect(gpu_time_sql)
    cur = conn.cursor()


    ###### get the StartLog time (1st cudaDeviceSynchronize) ######
    cur.execute('''
    WITH StartLog AS (
        SELECT MIN(start) AS first_start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%cudaDeviceSynchronize%'
    )
    SELECT first_start FROM StartLog;
    ''')
    first_start = cur.fetchone()[0]


    ###### all the traced kernel calls ######
    cur.execute(f'''
    SELECT k.start, k.end
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE k.start > {first_start};
    ''')

    kernels_time = cur.fetchall()
    for i in range(len(kernels_time)):
        kernels_time[i] = kernels_time[i][0] / 1000, kernels_time[i][1] / 1000

    # cur.execute(f'''
    # SELECT k.start, (k.end - k.start) AS exe, s.value AS name
    # FROM CUPTI_ACTIVITY_KIND_KERNEL k
    # JOIN StringIds s ON k.demangledName = s.id
    # WHERE k.start > {first_start};
    # ''')

    # with open('cuda_kernel_calls.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['start', 'exe', 'name'])
    #     for row in cur.fetchall():
    #         writer.writerow(row)


    ###### all the traced memcpy calls ######
    cur.execute(f'''
    SELECT (m.end - m.start) AS exe
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    WHERE m.start > {first_start};
    ''')

    memcpys = cur.fetchall()
    for i in range(len(memcpys)):
        memcpys[i] = memcpys[i][0] / 1000

    # with open('cuda_memcpy_calls.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['start', 'exe'])
    #     for row in cur.fetchall():
    #         writer.writerow(row)


    conn.close()
    conn = sqlite3.connect(kernel_wrapper_sql)
    cur = conn.cursor()


    ###### group the kernels within each kernel wrapper ######
    cur.execute('''
    WITH StartLog AS (
        SELECT MIN(start) AS first_start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE '%cudaDeviceSynchronize%'
    )
    SELECT s.value AS name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE r.start > (SELECT first_start FROM StartLog);
    ''')

    apis = cur.fetchall()
    kernels = []
    start, end = 0, 0
    for api in apis:
        if 'cudaLaunchKernel' in api[0]:
            if start == 0:
                start = kernels_time[0][0]
            end = kernels_time[0][1]
            kernels_time.pop(0)
        elif 'cudaDeviceSynchronize' in api[0]:
            kernels.append(end - start)
            start, end = 0, 0
    assert kernels_time == []

    conn.close()
    return kernels, memcpys
