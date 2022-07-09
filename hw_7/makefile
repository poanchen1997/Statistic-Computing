RANDOM_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas
RANDOM_INCLUDE = -I/usr/include/
RANDOM_LIB = -L/usr/lib/
RANDOM_OBJS = random

all:	${RANDOM_OBJS}
	rm -f *.o

random.o: random.cpp random.h
	gcc -g -c random.cpp -o random.o ${RANDOM_INCLUDE}

main.o: main.cpp random.h
	gcc -g -c main.cpp -o main.o

random: main.o random.o
	gcc main.o random.o -o random ${RANDOM_LIB} ${RANDOM_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${RANDOM_OBJS}