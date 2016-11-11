# Makefile for 159.735 Assignment 3
#

CPP = g++

# Use this for your CUDA programs
NVCC = nvcc

# FLAGS for Linux
CFLAGS = -w -O3

# Locally compiled modules
OBJS = fitsfile.o lenses.o

# Link to CFITSIO libraries - modify these accordingly
LIBP = -L/home/s16242950/cfitsio/
INCP = -I/home/s16242950/cfitsio/

LIBS = -lcfitsio -lm

MODS = $(INCP) $(LIBP) $(LIBS) $(OBJS) 

BINS = lens_seq

all : $(BINS)

clean :
	rm -f $(BINS)
	rm -f *.o

# Demo program. Add more programs by making entries similar to this
lens_seq : lens_seq.cpp $(OBJS)
	${CPP} $(CFLAGS) -o lens_seq lens_seq.cpp $(MODS)

# Modules compiled and linked separately
fitsfile.o : fitsfile.cpp fitsfile.h
	${CPP} $(CFLAGS) $(INCP) -c fitsfile.cpp

lenses.o : lenses.cpp lenses.h
	${CPP} $(CFLAGS) $(INCP) -c lenses.cpp

