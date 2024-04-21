#This is a comment
INC="./inc"
FLAGS=-I$(INC)
OMPFLAGS=-fopenmp
CFLAGS = -g -Wall
CC=gcc
CPP=g++
	
#SOURCES=$(wildcard *.c)
OBJECTS=$(patsubst %.c, %, $(wildcard *.c)) $(patsubst %.cpp, %, $(wildcard *.cpp))


all: $(OBJECTS)
	@echo 'objects are "$(OBJECTS)"'

%: %.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

%: %.cpp
	$(CPP) $(CFLAGS) $(OMPFLAGS) $< -o $@
	
clean:
	rm -vf $(OBJECTS)
