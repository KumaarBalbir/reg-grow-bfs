# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = `pkg-config --cflags --libs opencv4` -Wall -O2

# Targets
TARGET1 = reg_grow
TARGET2 = reg_grow_dir

# Source files
SRC1 = reg_grow.cpp
SRC2 = reg_grow_dir.cpp

# Object files
OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cpp=.o)

# Default target: Compile both programs
all: $(TARGET1) $(TARGET2)

# Compile reg_grow
$(TARGET1): $(OBJ1)
	$(CXX) -o $(TARGET1) $(OBJ1) $(CXXFLAGS)

# Compile reg_grow_dir
$(TARGET2): $(OBJ2)
	$(CXX) -o $(TARGET2) $(OBJ2) $(CXXFLAGS)

# Clean object files and binaries
clean:
	rm -f $(OBJ1) $(TARGET1) $(OBJ2) $(TARGET2)

# PHONY targets
.PHONY: all clean
