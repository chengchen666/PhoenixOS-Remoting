import re
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py <input_filename> <output_filename>")
    sys.exit(1)

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename, 'r') as file:
    log_data = file.read()

pattern = re.compile(
    r'\[.*?\] \[.*?\] (\w+)\n\[.*?\] exe: ([\d.]+)',
)

matches = pattern.findall(log_data)

with open(output_filename, 'w') as outfile:
    for match in matches:
        if match[0][0] == "_":
            continue
        outfile.write(f"{match[0]}, {match[1]}\n")
