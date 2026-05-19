CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SRCS := $(wildcard src/*.cpp)
OBJS := $(SRCS:.cpp=.o)
BIN  := ctg

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(BIN) $(BIN).exe

.PHONY: clean
