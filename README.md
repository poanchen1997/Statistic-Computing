# Statistic-Computing

This course is designed for improving our code speed by using low level program language, in this course, we start from using R to code, and then using C/C++ to do the same things and compare the difference.

I put all our homework in this folder, because I think this is useful when I want to look up for some tips.

1. This homework is not hard, we just basically follow the formula to code and get what we want.

2. This homework asked us to code the AIC by hand, and then using this to do forward search and backward search to decide which variables is good for us. Finally evaluate the answer from two algorithm.

3. Different from the above, we this time try to use $MC^3$ (Markov chain Monte Carlo model composition) to decide which variables we should choose. This algorithm is way more complex than previous !

4. In this homework, I did the Newton-Raphson algorithm, Laplace Approximation and Metropolis-Hasting Algorithm. Besides, we use the "snow" package to implement the parallel computing, so we can reduce the computing time.

5. In this homework, we are asked to calculate the same value as homework 2, but this time, we need to use C/C++ to finish it. Also, we are asking to do it by two method, first is normal method, use the build-in array to do the matrix product, while the other one, we have to use the package "GSL" from fortran.

6. This homework first ask us to get the determent by recursion, and then compute value we want and store it by the link-list data structure.

7. In this homework, we use C/C++ to code a function to generate samples for multi-variate normal distribution.

8. This is the final exam of our course, in this project we use C/C++ to calculate the same thing as we do in hw4, also, we use "MPI" to do the parallel computing in this project.

-------------------------------------------------------------------------------

In the project using C/C++ to code, I use the above package :

* GSL - very useful for matrix computing
* CLAPACK - doing algebra in C/C++
* CBLAS - doing algebra in C/C++
* MPI - for parallel computing
* valgrind - to check the memory leak

