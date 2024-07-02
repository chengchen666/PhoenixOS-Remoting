## Get number of launchkernel apis

```sql
SELECT s.value AS api_name, COUNT(*) AS count
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value LIKE '%launchkernel%'
GROUP BY s.value;
```


## Get number of kernels executed

```sql
SELECT COUNT(*) AS kernel_count
FROM CUPTI_ACTIVITY_KIND_KERNEL;
```


## Output all traced cuda apis

```sql
.headers ON
.mode csv
.output cuda_api_calls.csv

WITH StartLog AS (
    SELECT MIN(start) AS first_start
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE s.value LIKE '%cudaDeviceSynchronize%'
)
SELECT r.start, (r.end - r.start) AS exe, s.value AS name
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start > (SELECT first_start FROM StartLog);

.output stdout
```


## Output all traced kernel executions

```sql
.headers ON
.mode csv
.output cuda_kernel_calls.csv

WITH StartLog AS (
    SELECT MIN(start) AS first_start
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE s.value LIKE '%cudaDeviceSynchronize%'
)
SELECT k.start, (k.end - k.start) AS exe, s.value AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE k.start > (SELECT first_start FROM StartLog);

.output stdout
```


## Output all traced memcpy executions

```sql
.headers ON
.mode csv
.output cuda_memcpy_calls.csv

WITH StartLog AS (
    SELECT MIN(start) AS first_start
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE s.value LIKE '%cudaDeviceSynchronize%'
)
SELECT m.start, (m.end - m.start) AS exe
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
WHERE m.start > (SELECT first_start FROM StartLog);

.output stdout
```
