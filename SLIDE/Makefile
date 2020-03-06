TARGET := runme

CPPOBJDIR = cppobjs

CPPSOURCES := $(wildcard *.cpp)
CPPOBJS := $(CPPSOURCES:%.cpp=$(CPPOBJDIR)/%.o)

COBJDIR = cobjs

CSOURCES := $(wildcard *.c)
COBJS := $(CSOURCES:%.c=$(COBJDIR)/%.o)

OBJS = $(CPPOBJS) $(COBJS)

CXX := g++-6
CC := gcc-6
RM := rm

#OPT_FLAGS   := -fno-strict-aliasing -O2 -fopenmp #-march=skylake -mtune=intel -mavx2 -mavx512f
OPT_FLAGS   := -fno-strict-aliasing -g -O3 -fopenmp -march=skylake -mtune=intel -mavx2 -mavx512f

INC := /usr/include/

CNPY := ~/workspace/baseline/cnpy

LIB += -fPIC -fopenmp -L $(CNPY)/build/ -lcnpy -lz

CXXFLAGS := -m64  -DUNIX -I $(CNPY) -lcnpy -lz -std=c++11 $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)
CFLAGS := -m64 -DUNIX -I $(CNPY) -lcnpy -lz $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)

LDFLAGS := $(LIBRARY_PATH) $(LIB) -Wl,-rpath=$(CNPY)/build

.PHONY: clean deps

$(TARGET): $(CPPOBJDIR) $(COBJDIR) $(CPPOBJS) $(COBJS)
	$(CXX) -o $(TARGET) $(CPPOBJS) $(LDFLAGS)

$(CPPOBJS): $(CPPOBJDIR)/%.o: %.cpp
	@echo "compile $@ $<"
	$(CXX) -fPIC $(CXXFLAGS) -c $< -o $@
        
$(COBJS): $(COBJDIR)/%.o: %.c
	@echo "compile $@ $<"
	$(CC) -fPIC $(CFLAGS) -c $< -o $@
        
$(CPPOBJDIR):   
	@mkdir -p $(CPPOBJDIR)
        
$(COBJDIR):     
	@mkdir -p $(COBJDIR)

clean:
	$(RM) -f $(TARGET) $(OBJ)
	$(RM) -rf $(CPPOBJDIR)
	$(RM) -rf $(COBJDIR)

deps:
	# fetch and build cnpy
	rm -rf deps
	mkdir deps
	(cd deps; git clone https://github.com/rogersce/cnpy.git)
	(cd deps/cnpy; mkdir build; cd build; cmake ../../cnpy; make)
