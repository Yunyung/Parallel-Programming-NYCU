CXX := g++
CXXFLAGS = -I./common -O3 -std=c++17 -Wall -lpthread
all: pi

pi: pi.cpp
	$(CXX) pi.cpp $(CXXFLAGS) -o ./pi.out

.PHONY : clean

clean:
	rm -f *.out pi