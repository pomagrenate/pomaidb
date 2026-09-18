# Makefile for chaos_anomaly_detector
# This assumes pomaidb is already built in the parent directory

POMAIDB_ROOT = ../..
BUILD_DIR = $(POMAIDB_ROOT)/build
SRC_DIR = $(POMAIDB_ROOT)/src
INCLUDE_DIR = $(POMAIDB_ROOT)/include
CAPPI_DIR = $(POMAIDB_ROOT)/src/capi
UTILS_DIR = $(POMAIDB_ROOT)/src/utils
PALLOC_DIR = $(POMAIDB_ROOT)/third_party/palloc/include

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(CAPPI_DIR) -I$(UTILS_DIR) -I$(PALLOC_DIR)
LDFLAGS = -L$(BUILD_DIR) -lpomai -lpomai_c -lpalloc-static -lpthread

# Windows detection
ifeq ($(OS),Windows_NT)
    CXX = g++
    LDFLAGS = -L$(BUILD_DIR) -lpomai -lpomai_c -lpalloc-static -lws2_32
    EXE_EXT = .exe
else
    LDFLAGS = -L$(BUILD_DIR) -lpomai -lpomai_c -lpalloc-static -lpthread
    EXE_EXT =
endif

TARGET = chaos_anomaly_detector$(EXE_EXT)
SOURCE = chaos_anomaly_detector.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	@echo "Building chaos_anomaly_detector..."
	@echo "Make sure pomaidb is built first in: $(BUILD_DIR)"
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -rf chaos_anomaly_db

# Run the chaos test
run: $(TARGET)
	./$(TARGET)

# Build pomaidb first (helper target)
build-pomaidb:
	cd $(POMAIDB_ROOT) && mkdir -p build && cd build && cmake .. && cmake --build .

.PHONY: all clean run build-pomaidb
