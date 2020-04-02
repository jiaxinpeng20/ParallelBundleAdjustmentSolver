CXX = mpicxx	
WORK_DIR = .



CPPFLAGS = -I$(WORK_DIR)/eigen -I/home/jiaxin/Public/ezvs_deps2015/include/
LDFLAGS =  -L/home/jiaxin/Public/ezvs_deps2015/lib/ -L/home/jiaxin/Public/ezvs_deps2015/lib64/ -lboost_mpi  -lboost_serialization -O2


OBJECTS = driver.o ParallelBundleAdjustmentSolver.o

driver: $(OBJECTS)
	$(CXX) -o driver $(OBJECTS) $(CPPFLAGS) $(LDFLAGS)
driver.o: driver.cpp
	$(CXX) -c driver.cpp $(CPPFLAGS)
ParallelBundleAdjustmentSolver.o: ParallelBundleAdjustmentSolver.cpp
	$(CXX) -c ParallelBundleAdjustmentSolver.cpp $(CPPFLAGS) $(LDFLAGS)
clean:
	rm driver driver.o ParallelBundleAdjustmentSolver.o
