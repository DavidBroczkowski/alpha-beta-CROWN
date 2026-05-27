# Change the path based on your installation folder.
CPX_PATH ?= /opt/ibm/ILOG/CPLEX_Studio2211/cplex/

CC = g++
ARCH = x86-64_linux
STANDARD = -std=c++17

cpx:
	$(CC) -g $(STANDARD) -L $(CPX_PATH)/lib/$(ARCH)/static_pic/ -L lib/ -I $(CPX_PATH)/include/ilcplex/ -o get_cuts get_cuts.cpp -lm -lcplex -ldl -lpthread -Wall
