import re
import sys

# 检查命令行参数
if len(sys.argv) != 3:
    print("Usage: python script.py <input_filename> <output_filename>")
    sys.exit(1)

input_filename = sys.argv[1]
output_filename = sys.argv[2]

# 打开并读取日志文件
with open(input_filename, 'r') as file:
    log_data = file.read()

# 定义正则表达式模式
pattern = re.compile(
    r'\[.*?\] (\d+), \[.*?\] (\w+)\n\[.*?\] , (\d+)'
)

# 在数据中查找匹配项
matches = pattern.findall(log_data)

# 打开输出文件写入结果
with open(output_filename, 'w') as outfile:
    for match in matches:
        outfile.write(f"{match[0]}, {match[1]}, {match[2]}\n")
