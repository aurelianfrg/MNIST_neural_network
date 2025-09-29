// MNIST_neural_network.cpp : définit le point d'entrée de l'application.


#include <iostream>

#include "mnist_reader_less.hpp"
#include "gpu_matrix_op.hpp"
#include "neural_network.hpp"

using namespace std;

void printMnistDigit(uint8_t* img, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (img[i * size + j] > 128) {
                cout << "* ";
            }
            else {
                cout << "  ";
            }
        }
        cout << endl;
		}
}

int main()
{
    const GLuint width_right = 2;
    const GLuint height_left = 3;
    const GLuint common_length = 1;

    // --- Initialize GLFW + OpenGL ---
    init_gl();


	// --- Load MNIST dataset ---
	auto reader = mnist::read_dataset<uint8_t, uint8_t>();

    //visualisation
	printMnistDigit(reader.training_images[0].data(), 28);
	cout << "Label: " << int(reader.training_labels[0]) << endl;

	// --- test matrix multiplication ---

	/*GLuint ssboResult;
	glGenBuffers(1, &ssboResult);

    GLdouble *mat1 = new GLdouble[common_length * height_left];
    GLdouble *mat2 = new GLdouble[width_right * common_length];
    fillMatrix<GLdouble>(mat1, common_length, height_left);
    fillMatrix<GLdouble>(mat2, width_right, common_length);

    printMatrix<GLdouble>(mat1, common_length, height_left);
    std::cout << std::endl;
    printMatrix<GLdouble>(mat2, width_right, common_length);
    std::cout << std::endl;
  

    matrix_add<GLdouble>(mat1, mat1, ssboResult, common_length, height_left);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
	GLdouble* result = (GLdouble*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

	cout << "Result:" << endl;
    printMatrix<GLdouble>(result, common_length, height_left);
    cout << endl;

    glDeleteBuffers(1, &ssboResult);
    delete[] mat1;
    delete[] mat2;*/


	// --- test neural network ---
	
	const int inputSize = 15;

    float input_img[inputSize] = {
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.85,
        0.6, 0.7, 0.8, 0.3, 0.4
	};
	float input_vec[3] = { 0.1, 0.2, 0.3 };
	GLuint ssboResult;
	glGenBuffers(1, &ssboResult);
	matrix_add_constant_vec<float>(input_img, input_vec, ssboResult, inputSize/3, 3);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
	float* result = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	printMatrix<float>(result, inputSize/3, 3);

    // for each layer, number of columns (width) must match size of input (number of pixels for first layer / previous number of neurons)
	// and number of rows (height) is the number of neurons in the layer
    vector<unsigned int> neuronsPerLayer({ 10, 3 });
    NeuralNetwork<float> nn(2, neuronsPerLayer, inputSize);

    cout << "Neural Network initialized:" << endl;
    cout << nn << endl;

	// 1 image is a single vector of width=1 and height=width*height
	// the width of the input matrix corresponds to the number of images

	vector<float> output = nn.feedForward(input_img, 1, inputSize);
	cout << "Neural Network output:" << endl;
    for (float & val : output) {
        cout << std::fixed << std::setprecision(2) << val << " ";
	}
    cout << endl;

    float expected[3] = { 0.0f, 0.0f, 1.0f };
	nn.backPropagation(expected, 1);


    cout << endl;
	return 0;
}
