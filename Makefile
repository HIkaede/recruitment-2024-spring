CXXFLAGS=-O1 -Wall -Wextra -Wshadow -pipe -fopenmp

all: main

main: baseline.o main.o solution.o
	g++ $(CXXFLAGS) -o $@ $^

baseline.o: baseline.cc
	g++ -O0 -Wall -Wextra -Wshadow -pipe -c -o $@ $^

%.o: %.cc
	g++ $(CXXFLAGS) -c -o $@ $^

clean:
	rm -f *.o main
