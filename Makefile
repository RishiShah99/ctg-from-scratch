CXX      := g++
CXXFLAGS := -std=c++17 -O1 -Wall -Wextra
DEMOFL   := -std=c++17 -O0 -Wall -Wextra
LDFLAGS  := -Wl,--stack,268435456

COMMON_SRC := src/value.cpp src/nn.cpp src/data.cpp src/loss.cpp src/optim.cpp
CTG_SRC    := $(COMMON_SRC) src/tabm.cpp src/tx.cpp src/main.cpp
DEMO_SRC   := $(COMMON_SRC) src/demo.cpp
CGM_SRC    := src/value.cpp src/loss.cpp src/optim.cpp src/complex_value.cpp src/s4d.cpp src/osdn.cpp src/cgm_smoke.cpp
CGM_DEMO_SRC := src/value.cpp src/complex_value.cpp src/s4d.cpp src/osdn.cpp src/cgm_data.cpp src/cgm_demo.cpp
CGM_TRAIN_SRC := src/value.cpp src/loss.cpp src/optim.cpp src/complex_value.cpp src/s4d.cpp src/osdn.cpp src/cgm_data.cpp src/cgm_train.cpp
CGM_INFTEST_SRC := src/value.cpp src/osdn.cpp src/osdn_inference_test.cpp
FEATTEST_SRC    := esp32/src/features.cpp src/features_host_test.cpp
BLOB2H_SRC      := src/blob_to_header.cpp
NORMSTATS_SRC   := src/cgm_data.cpp tools/extract_norm_stats.cpp

CTG          := ctg
DEMO         := demo
CGM_SMOKE    := cgm_smoke
CGM_DEMO     := cgm_demo
CGM_TRAIN    := cgm_train
CGM_INFTEST  := cgm_inference_test
FEATTEST     := features_host_test
BLOB2H       := blob_to_header
NORMSTATS    := extract_norm_stats

all: $(CTG) $(DEMO)

$(CTG): $(CTG_SRC)
	$(CXX) $(CXXFLAGS) $(CTG_SRC) -o $@ $(LDFLAGS)

$(DEMO): $(DEMO_SRC)
	$(CXX) $(DEMOFL) $(DEMO_SRC) -o $@ $(LDFLAGS)

$(CGM_SMOKE): $(CGM_SRC)
	$(CXX) $(CXXFLAGS) $(CGM_SRC) -o $@ $(LDFLAGS)

$(CGM_DEMO): $(CGM_DEMO_SRC)
	$(CXX) $(DEMOFL) $(CGM_DEMO_SRC) -o $@ $(LDFLAGS)

$(CGM_TRAIN): $(CGM_TRAIN_SRC)
	$(CXX) $(DEMOFL) $(CGM_TRAIN_SRC) -o $@ $(LDFLAGS)

$(CGM_INFTEST): $(CGM_INFTEST_SRC) src/osdn_inference.h
	$(CXX) $(DEMOFL) $(CGM_INFTEST_SRC) -o $@ $(LDFLAGS)

$(FEATTEST): $(FEATTEST_SRC) esp32/src/features.h
	$(CXX) $(CXXFLAGS) $(FEATTEST_SRC) -o $@ $(LDFLAGS)

$(BLOB2H): $(BLOB2H_SRC)
	$(CXX) $(CXXFLAGS) $(BLOB2H_SRC) -o $@ $(LDFLAGS)

$(NORMSTATS): $(NORMSTATS_SRC) src/cgm_data.hpp
	$(CXX) $(DEMOFL) $(NORMSTATS_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f src/*.o \
	      $(CTG) $(CTG).exe \
	      $(DEMO) $(DEMO).exe \
	      $(CGM_SMOKE) $(CGM_SMOKE).exe \
	      $(CGM_DEMO) $(CGM_DEMO).exe \
	      $(CGM_TRAIN) $(CGM_TRAIN).exe \
	      $(CGM_INFTEST) $(CGM_INFTEST).exe \
	      $(FEATTEST) $(FEATTEST).exe \
	      $(BLOB2H) $(BLOB2H).exe \
	      $(NORMSTATS) $(NORMSTATS).exe

.PHONY: all clean
