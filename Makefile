# Makefile for HNSW K-Nearest Neighbors

CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
TARGET = hnsw_knn
TEST_TARGET = hnsw_test
SOURCES = main.cpp HNSWIndex.cpp
TEST_SOURCES = test.cpp HNSWIndex.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
HEADERS = HNSWIndex.h

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): main.o HNSWIndex.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.o HNSWIndex.o
	@echo "Build complete! Run with: ./$(TARGET)"

# Build test executable
$(TEST_TARGET): test.o HNSWIndex.o
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) test.o HNSWIndex.o
	@echo "Test build complete! Run with: ./$(TEST_TARGET)"

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TARGET) $(TEST_TARGET) test.o
	@echo "Clean complete!"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Build and run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Rebuild from scratch
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  make         - Build the main program"
	@echo "  make run     - Build and run the main program"
	@echo "  make test    - Build and run test suite"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make rebuild - Clean and rebuild"
	@echo "  make help    - Show this help message"

.PHONY: all clean run test rebuild help