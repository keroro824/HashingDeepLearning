TARGET  := runme

CPPOBJDIR = cppobjs

CPPSOURCES := $(wildcard *.cpp)
CPPOBJS := $(CPPSOURCES:%.cpp=$(CPPOBJDIR)/%.o)

COBJDIR = cobjs

CSOURCES := $(wildcard *.c)
COBJS := $(CSOURCES:%.c=$(COBJDIR)/%.o)

OBJS = $(CPPOBJS) $(COBJS)

OPT_FLAGS   := -fno-strict-aliasing -O2 -fopenmp

INC := /usr/include/

LIB += -fopenmp -L /Users/beidchen/Documents/work/cnpy/build/ -lcnpy -lz

CXXFLAGS := -m64 -DUNIX -L /Users/beidchen/Documents/work/cnpy/build/ -lcnpy -lz -std=c++11 $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)
CFLAGS := -m64 -DUNIX -L /Users/beidchen/Documents/work/cnpy/build/ -lcnpy -lz $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)

LDFLAGS := $(LIBRARY_PATH) $(LIB)

.PHONY: clean

$(TARGET): $(CPPOBJDIR) $(COBJDIR) $(CPPOBJS) $(COBJS)
	g++-5 -fPIC -L /Users/beidchen/Documents/work/cnpy/build/ -lcnpy -lz -o $(TARGET) $(CPPOBJS) $(LDFLAGS)

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
