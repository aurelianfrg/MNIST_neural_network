// MNIST_neural_network.cpp : définit le point d'entrée de l'application.


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

float* setup_training_set(mnist::MNIST_dataset<uint8_t, uint8_t> & reader, const unsigned int inputSize, const unsigned int inputsNumber) {
    // to be used as input to the neural network, the rows must represent the pixels of an image and the columns the different images
    // using row-major storage, we need to transpose the dataset
    // this means that in memory, all the first pixels of each image will be stored in a row, then the second pixels ...

    float* training_set = new float[inputSize * inputsNumber];
    for (int input = 0; input < inputsNumber; ++input) {
        vector<uint8_t> img = reader.training_images[input];
        for (int pixel_rank = 0; pixel_rank < inputSize; ++pixel_rank) {
            float normalised_pixel = img[pixel_rank] / 255.f;
            training_set
        }
    }
}

void MNIST_neural_network_training() {

    // Setup MNIST dataset to be a valid input for the neural network : a contiguous array of floats or double
    mnist::MNIST_dataset<uint8_t, uint8_t> reader = mnist::read_dataset<uint8_t, uint8_t>();

    const unsigned int inputSize = 28 * 28; // size of input layer = number of pixels in an image
    const unsigned int inputsNumber = 10000;

	// to be used as input to the neural network, the rows must represent the pixels of an image and the columns the different images
	// using row-major storage, we need to transpose the dataset
    // this means that in memory, all the first pixels of each image will be stored in a row, then the second pixels ...

   

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
    const unsigned int outputSize = 2;
    const unsigned int inputNumber = 2;


    float input[inputNumber *inputSize] = {
        0.1, 0.8,
        0.9, 0.8,
        0.1, 0.2
	};
	
    // for each layer, number of columns (width) must match size of input (number of pixels for first layer / previous number of neurons)
	// and number of rows (height) is the number of neurons in the layer
    vector<unsigned int> neuronsPerLayer({ 4, outputSize });
    NeuralNetwork<float> nn(2, neuronsPerLayer, inputSize);

    cout << "Neural Network initialized:" << endl;
    cout << nn << endl;

	// 1 image is a single vector of width=1 and height=width*height
	// the width of the input matrix corresponds to the number of images

	/*vector<float> output = nn.feedForward(input_img, 1, inputSize);
	cout << "Neural Network output:" << endl;
    for (float & val : output) {
        cout << std::fixed << std::setprecision(2) << val << " ";
	}*/
    cout << endl;

	float learningRate = 0.02f;
    float expected[inputNumber*2] = { 
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    // real training
	//nn.setVerbose(true);
    nn.train(input, expected, inputNumber, 2000, learningRate);
	cout << "Neural Network after training:" << endl << nn << endl;
        
    vector<float> finalAnswer = nn.feedForward(input, 2, 3);
	cout << "Neural Network output after training:" << endl;
	printMatrix<float>(finalAnswer.data(), inputNumber, outputSize);

    cout << endl;
	return 0;
}


// TODO : store all data in gl buffers to avoid cpu-gpu transfers
// TODO : coherence of height and width variables (sometimes swapped)
