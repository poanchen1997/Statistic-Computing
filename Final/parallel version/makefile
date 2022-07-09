FINAL_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas 
FINAL_INCLUDE = -I/usr/include/
FINAL_LIB = -L/usr/lib/atlas
FINAL_OBJS = final

all:	${FINAL_OBJS}
	rm -f *.o

final.o: final.cpp final.h
	mpic++ -g -c final.cpp -o final.o ${FINAL_INCLUDE}

main.o: main.cpp final.h
	mpic++ -g -c main.cpp -o main.o 

final: main.o final.o
	mpic++ main.o final.o -o final ${FINAL_LIB} ${FINAL_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${FINAL_OBJS}