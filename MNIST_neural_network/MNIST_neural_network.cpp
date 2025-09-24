// MNIST_neural_network.cpp : définit le point d'entrée de l'application.


#include <iostream>

#include "mnist_reader_less.hpp"
#include "gpu_matrix_op.hpp"
#include "neural_network.hpp"

using namespace std;

int main()
{
    const GLuint size = 5;

    // --- Initialize GLFW + OpenGL ---
    init_gl();
	auto reader = mnist::read_dataset<uint8_t, uint8_t>();

	// --- test matrix multiplication ---
	GLuint ssboResult;
	glGenBuffers(1, &ssboResult);

    GLdouble *mat1 = new GLdouble[size * size];
    GLdouble *mat2 = new GLdouble[size * size];
    fillMatrix<GLdouble>(mat1, size);
    fillMatrix<GLdouble>(mat2, size);

    if (size <= 64) {
        printMatrix<GLdouble>(mat1, size);
        std::cout << std::endl;
        printMatrix<GLdouble>(mat2, size);
        std::cout << std::endl;
    }

    matrix_add<GLdouble>(mat1, mat2, ssboResult, size, size, size);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
	GLdouble* result = (GLdouble*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    if (size <= 64) {
        printMatrix<GLdouble>(result, size);
        cout << endl;
    }
    glDeleteBuffers(1, &ssboResult);
    delete[] mat1;
    delete[] mat2;


	// --- test neural network ---
	int layersWidths[2] = { 5, 5 };
	int layersHeights[2] = { 3, 3 };
	NeuralNetwork<2, double> nn(layersWidths, layersHeights);

	cout << "Neural Network initialized:" << endl;
	cout << nn << endl;


	return 0;

}
