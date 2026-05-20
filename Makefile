CXX      := g++
CXXFLAGS := -std=c++17 -O1 -Wall -Wextra
DEMOFL   := -std=c++17 -O0 -Wall -Wextra
LDFLAGS  := -Wl,--stack,268435456

COMMON_SRC := src/value.cpp src/nn.cpp src/data.cpp src/loss.cpp src/optim.cpp
CTG_SRC    := $(COMMON_SRC) src/tabm.cpp src/tx.cpp src/main.cpp
DEMO_SRC   := $(COMMON_SRC) src/demo.cpp

CTG  := ctg
DEMO := demo

all: $(CTG) $(DEMO)

$(CTG): $(CTG_SRC)
	$(CXX) $(CXXFLAGS) $(CTG_SRC) -o $@ $(LDFLAGS)

$(DEMO): $(DEMO_SRC)
	$(CXX) $(DEMOFL) $(DEMO_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f src/*.o $(CTG) $(CTG).exe $(DEMO) $(DEMO).exe

.PHONY: all clean
