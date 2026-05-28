CXX      := g++
CXXFLAGS := -std=c++17 -O1 -Wall -Wextra
# DEMOFL: -O0 retained for trainer/test/norm-stats binaries where
# debuggability under gdb matters more than wall-clock speed.
DEMOFL   := -std=c++17 -O0 -Wall -Wextra
# SHOWFL: -O2 for user-facing demo binaries (DEMO, CGM_DEMO). These
# render inference timings to the terminal and pitched the project's
# "fast on-device inference" story; building them at -O0 was off-message
# by 5-10×. No correctness-sensitive binary uses this flag set.
SHOWFL   := -std=c++17 -O2 -Wall -Wextra
LDFLAGS  := -Wl,--stack,268435456

LOCKED_WEIGHTS  := results/osdn-brain-ohio-clamped.weights.bin.best
LOCKED_BITIDENT := 7.96795993e-07

COMMON_SRC := src/value.cpp src/nn.cpp src/data.cpp src/loss.cpp src/optim.cpp
CTG_SRC    := $(COMMON_SRC) src/tabm.cpp src/tx.cpp src/main.cpp
DEMO_SRC   := $(COMMON_SRC) src/demo.cpp
CGM_SRC    := src/value.cpp src/loss.cpp src/optim.cpp src/complex_value.cpp src/s4d.cpp src/osdn.cpp src/cgm_smoke.cpp
CGM_DEMO_SRC := src/value.cpp src/complex_value.cpp src/s4d.cpp src/osdn.cpp src/cgm_data.cpp src/cgm_demo.cpp
CGM_TRAIN_SRC := src/value.cpp src/loss.cpp src/optim.cpp src/complex_value.cpp src/s4d.cpp src/osdn.cpp src/cgm_data.cpp src/cgm_train.cpp
CGM_INFTEST_SRC := src/value.cpp src/osdn.cpp src/osdn_inference_test.cpp
FEATTEST_SRC    := esp32/src/features.cpp src/features_host_test.cpp
GRAD_CHECK_SRC  := src/value.cpp src/grad_check.cpp
BLOB2H_SRC      := src/blob_to_header.cpp
NORMSTATS_SRC   := src/cgm_data.cpp tools/extract_norm_stats.cpp

CTG          := ctg
DEMO         := demo
CGM_SMOKE    := cgm_smoke
CGM_DEMO     := cgm_demo
CGM_TRAIN    := cgm_train
CGM_INFTEST  := cgm_inference_test
FEATTEST     := features_host_test
GRAD_CHECK   := grad_check
BLOB2H       := blob_to_header
NORMSTATS    := extract_norm_stats

all: $(CTG) $(DEMO)

$(CTG): $(CTG_SRC)
	$(CXX) $(CXXFLAGS) $(CTG_SRC) -o $@ $(LDFLAGS)

$(DEMO): $(DEMO_SRC)
	$(CXX) $(SHOWFL) $(DEMO_SRC) -o $@ $(LDFLAGS)

$(CGM_SMOKE): $(CGM_SRC)
	$(CXX) $(CXXFLAGS) $(CGM_SRC) -o $@ $(LDFLAGS)

$(CGM_DEMO): $(CGM_DEMO_SRC)
	$(CXX) $(SHOWFL) $(CGM_DEMO_SRC) -o $@ $(LDFLAGS)

$(CGM_TRAIN): $(CGM_TRAIN_SRC)
	$(CXX) $(DEMOFL) $(CGM_TRAIN_SRC) -o $@ $(LDFLAGS)

$(CGM_INFTEST): $(CGM_INFTEST_SRC) src/osdn_inference.h
	$(CXX) $(DEMOFL) $(CGM_INFTEST_SRC) -o $@ $(LDFLAGS)

$(FEATTEST): $(FEATTEST_SRC) esp32/src/features.h
	$(CXX) $(CXXFLAGS) $(FEATTEST_SRC) -o $@ $(LDFLAGS)

$(GRAD_CHECK): $(GRAD_CHECK_SRC) src/value.hpp
	$(CXX) $(CXXFLAGS) $(GRAD_CHECK_SRC) -o $@ $(LDFLAGS)

$(BLOB2H): $(BLOB2H_SRC)
	$(CXX) $(CXXFLAGS) $(BLOB2H_SRC) -o $@ $(LDFLAGS)

$(NORMSTATS): $(NORMSTATS_SRC) src/cgm_data.hpp src/brain_config.h
	$(CXX) $(DEMOFL) $(NORMSTATS_SRC) -o $@ $(LDFLAGS)

check: $(CGM_INFTEST) $(FEATTEST) $(GRAD_CHECK)
	@echo "== check 1/3: $(CGM_INFTEST) (locked bit-identity = $(LOCKED_BITIDENT)) =="
	@out=$$(./$(CGM_INFTEST) --H 16 --K 8 --D-in 7 --n-layers 1 --L 144 --seed 42 \
	    --load-weights $(LOCKED_WEIGHTS)) && \
	 echo "$$out" && \
	 echo "$$out" | grep -q "= $(LOCKED_BITIDENT)" || \
	   { echo "FAIL: bit-identity drift (expected $(LOCKED_BITIDENT))"; exit 1; }
	@echo
	@echo "== check 2/3: $(FEATTEST) =="
	@./$(FEATTEST)
	@echo
	@echo "== check 3/3: $(GRAD_CHECK) =="
	@./$(GRAD_CHECK)
	@echo
	@echo "== ALL CHECKS PASS =="

clean:
	rm -f src/*.o \
	      $(CTG) $(CTG).exe \
	      $(DEMO) $(DEMO).exe \
	      $(CGM_SMOKE) $(CGM_SMOKE).exe \
	      $(CGM_DEMO) $(CGM_DEMO).exe \
	      $(CGM_TRAIN) $(CGM_TRAIN).exe \
	      $(CGM_INFTEST) $(CGM_INFTEST).exe \
	      $(FEATTEST) $(FEATTEST).exe \
	      $(GRAD_CHECK) $(GRAD_CHECK).exe \
	      $(BLOB2H) $(BLOB2H).exe \
	      $(NORMSTATS) $(NORMSTATS).exe

.PHONY: all clean check
