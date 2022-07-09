DETERMINANT_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas
DETERMINANT_INCLUDE = -I/usr/include/
DETERMINANT_LIB = -L/usr/lib/
DETERMINANT_OBJS = determinants

all:	${DETERMINANT_OBJS}
	rm -f *.o

determinant.o: determinant.cpp determinant.h
	gcc -g -c determinant.cpp -o determinant.o ${DETERMINANT_INCLUDE}

main.o: main.cpp determinant.h
	gcc -g -c main.cpp -o main.o

determinants: main.o determinant.o
	gcc main.o determinant.o -o determinants ${DETERMINANT_LIB} ${DETERMINANT_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${DETERMINANT_OBJS}