// MNIST_neural_network.cpp : définit le point d'entrée de l'application.
//

#include <iostream>

#include "mnist_reader_less.hpp"
#include "matrix_mult.hpp"

using namespace std;
const GLuint size = 1024;

int main()
{
	GLuint ssboResult;
	glGenBuffers(1, &ssboResult);

    GLdouble(*mat1)[n] = new GLdouble[n][n];
    GLdouble(*mat2)[n] = new GLdouble[n][n];
    /*GLdouble mat1[n][n] = { 0 };
    GLdouble mat2[n][n] = { 0 };*/
    fillMatrix(mat1);
    fillMatrix(mat2);

    if (n <= 64) {
        printMatrix(mat1);
        std::cout << std::endl;
        printMatrix(mat2);
        std::cout << std::endl;
    }

    array<array<GLdouble, n>, n> result = matrix_mult<n>(mat1, mat2, ssboResult);

    if (n <= 64) {
        printMatrix(result);
    }
    glDeleteBuffers(1, &ssboResult);
    delete[] mat1;
    delete[] mat2;
	return 0;

}
