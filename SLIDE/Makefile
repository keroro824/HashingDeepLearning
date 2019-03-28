TARGET  := runme

CPPOBJDIR = cppobjs

CPPSOURCES := $(wildcard *.cpp)
CPPOBJS := $(CPPSOURCES:%.cpp=$(CPPOBJDIR)/%.o)

COBJDIR = cobjs

CSOURCES := $(wildcard *.c)
COBJS := $(CSOURCES:%.c=$(COBJDIR)/%.o)

OBJS = $(CPPOBJS) $(COBJS)

OPT_FLAGS   := -fno-strict-aliasing -O2 -fopenmp #-march=skylake -mtune=intel -mavx2 -mavx512f

INC := /usr/include/

LIB += -fopenmp -L ~/workspace/baseline/cnpy/build/ -lcnpy -lz

CXXFLAGS := -m64  -DUNIX -I ~/workspace/baseline/cnpy/ -lcnpy -lz -std=c++11 $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)
CFLAGS := -m64 -DUNIX -I ~/workspace/baseline/cnpy/ -lcnpy -lz $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)

LDFLAGS := $(LIBRARY_PATH) $(LIB)

.PHONY: clean

$(TARGET): $(CPPOBJDIR) $(COBJDIR) $(CPPOBJS) $(COBJS)
	g++-5 -fPIC -L ~/workspace/baseline/cnpy/build/ -lcnpy -lz -o $(TARGET) $(CPPOBJS) $(LDFLAGS)

$(CPPOBJS): $(CPPOBJDIR)/%.o: %.cpp
	@echo "compile $@ $<"
	g++-5 -fPIC $(CXXFLAGS) -c $< -o $@
        
$(COBJS): $(COBJDIR)/%.o: %.c
	@echo "compile $@ $<"
	gcc-5 -fPIC $(CFLAGS) -c $< -o $@
        
$(CPPOBJDIR):   
	@ mkdir -p $(CPPOBJDIR)
        
$(COBJDIR):     
	@ mkdir -p $(COBJDIR)

clean:
	$(RM) $(TARGET) $(OBJ)
	$(RM) -rf $(CPPOBJDIR)
	$(RM) -rf $(COBJDIR)
