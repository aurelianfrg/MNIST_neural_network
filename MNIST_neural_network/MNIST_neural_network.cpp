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

void MNIST_neural_network_training() {

    // Setup MNIST dataset to be a valid input for the neural network : a contiguous array of floats or double
    mnist::MNIST_dataset<uint8_t, uint8_t> reader = mnist::read_dataset<uint8_t, uint8_t>();

    const unsigned int inputSize = 28 * 28; // size of input layer = number of pixels in an image


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



	// --- test neural network ---
	
    const unsigned int inputSize = 3;
    const unsigned int inputNumber = 2;

    float input_img[inputNumber *inputSize] = {
        0.1, 0.8, 0.9,
        0.8, 0.1, 0.2
	};
	
    // for each layer, number of columns (width) must match size of input (number of pixels for first layer / previous number of neurons)
	// and number of rows (height) is the number of neurons in the layer
    vector<unsigned int> neuronsPerLayer({ 3, 3 });
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

	float learningRate = 0.1f;
    float expected[inputNumber*3] = { 
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    // real training
	nn.setVerbose(true);
    nn.train(input_img, expected, inputNumber, 1, learningRate);
	cout << "Neural Network after training:" << endl << nn << endl;
        
    cout << endl;
	return 0;
}


// TODO : store all data in gl buffers to avoid cpu-gpu transfers
// TODO : coherence of height and width variables (sometimes swapped)
