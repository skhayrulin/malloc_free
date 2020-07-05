CC = g++ -std=c++14 -Wall -pedantic -pthread
TARGET:=p_sort

#SRC := $(wildcard *.cpp)
SRC := p_sort.cpp ocl_helper.cpp
OBJ := p_sort.o ocl_helper.o

OCL_INC  = -I/usr/local/cuda/include/
OCL_LIB  = -L/usr/local/cuda/lib64/

LIBS := -lOpenCL

CXXFLAGS = $(CC)
all: CXXFLAGS += -O3
all: $(TARGET)

debug: CXXFLAGS += -ggdb -O0
debug: $(TARGET)

$(TARGET): $(OBJ)
	$(CXXFLAGS) $(OCL_LIB) -o $(TARGET) $(OBJ) $(LIBS)

%.o: %.cpp
	$(CXXFLAGS) $(OCL_INC) -I. -c -o "$@" "$<"