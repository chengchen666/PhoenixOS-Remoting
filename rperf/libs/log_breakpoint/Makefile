NVCC = nvcc

# Target shared library
TARGET = log_breakpoint.so

# Source file
SRC = log_breakpoint.cu

# Compiler flags
CFLAGS = -shared -Xcompiler -fPIC

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

# Clean rule
clean:
	rm -f $(TARGET)

.PHONY: all clean
