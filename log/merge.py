
import re
import sys

if len(sys.argv) != 4:
    print("Usage: python script.py <client_file> <server_file> <output_file>")
    sys.exit(1)

client_file = sys.argv[1]
server_file = sys.argv[2]
output_file = sys.argv[3]

def compare_files_with_two_pointers(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output_file:
        pointer1 = 0
        pointer2 = 0

        cli_line = file1.readlines()
        ser_line = file2.readlines()

        len1 = len(cli_line)
        len2 = len(ser_line)

        while pointer1 < len1 or pointer2 < len2:
            cli_parts = cli_line[pointer1].strip().split(",")
            cli_name = cli_parts[0] 
            ser_parts = ser_line[pointer2].strip().split(",") 
            ser_name = ser_parts[0]

            if cli_name != ser_name:
                output_file.write(cli_line[pointer1])
            else:
                output_file.write(cli_line[pointer1])
                output_file.write(ser_line[pointer2])
                pointer2 += 1

            pointer1 += 1

        if pointer1 < len1:
            for i in range(pointer1, len1):
                output_file.write(cli_line[i])
                output_file.write("\n")

compare_files_with_two_pointers(client_file, server_file, output_file)
