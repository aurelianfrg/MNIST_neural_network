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

float* setup_dataset(vector<vector<uint8_t>> & raw_set, const unsigned int inputSize, const unsigned int inputsNumber) {
    // to be used as input to the neural network, the rows must represent the pixels of an image and the columns the different images
    // using row-major storage, we need to transpose the dataset
    // this means that in memory, all the first pixels of each image will be stored in a row, then the second pixels ...

    float* dataset = new float[inputSize * inputsNumber];
    for (int input = 0; input < inputsNumber; ++input) {
        vector<uint8_t> & img = raw_set[input];
        for (int pixel_rank = 0; pixel_rank < inputSize; ++pixel_rank) {
            float normalized_pixel = img[pixel_rank] / 255.f;
            dataset[input + pixel_rank * inputsNumber] = normalized_pixel;
        }
    }
    return dataset;
}

float* setup_labels(vector<uint8_t> raw_labels, const unsigned int inputsNumber) {
    // to be used to train the neural network, labels must be presented as bitmaps of size 10 with a single bit being at 1 : the bit of rank equal to the digit value

    float* labels = new float[10 * inputsNumber];
    for (int input = 0; input < inputsNumber; ++input) {
        uint8_t label = raw_labels[input];
        for (int bit_rank = 0; bit_rank < 10; ++bit_rank) {
            labels[input + bit_rank * inputsNumber] = (bit_rank == label) ? 1:0;
        }
    }
    return labels;
}

void MNIST_neural_network_training() {

    // Setup MNIST dataset to be a valid input for the neural network : a contiguous array of floats or double
    mnist::MNIST_dataset<uint8_t, uint8_t> reader = mnist::read_dataset<uint8_t, uint8_t>();

    const unsigned int inputSize = 28 * 28; // size of input layer = number of pixels in an image
    const unsigned int inputsNumber = 2;

    //visualisation
    printMnistDigit(reader.training_images[0].data(), 28);
    cout << "Label: " << int(reader.training_labels[0]) << endl;

	// to be used as input to the neural network, the rows must represent the pixels of an image and the columns the different images
	// using row-major storage, we need to transpose the dataset
    // this means that in memory, all the first pixels of each image will be stored in a row, then the second pixels ...

    float* training_set = setup_dataset(reader.training_images, inputSize, inputsNumber);
    float* labels = setup_labels(reader.training_labels, inputsNumber);
	float* single_input = setup_dataset(reader.training_images, inputSize, 1);
	float* single_label = setup_labels(reader.training_labels, 1);

	printMatrix<float>(training_set, inputsNumber, inputSize);
	printMatrix<float>(labels, inputsNumber, 10);

    // setup the neural network    
    vector<unsigned int> neuronsPerLayer({ 10, 10 });
    NeuralNetwork<float> nn(2, neuronsPerLayer, inputSize);
	cout << "Neural Network initialized:" << endl;
	cout << nn << endl;
	//nn.setVerbose(true);

    // test on a single input
	vector<float> output = nn.feedForward(single_input, 1, inputSize);
	cout << "Neural Network output for a single input before training:" << endl;
	printMatrix<float>(output.data(), 1, 10);
	cout << "Expected output:" << endl;
	printMatrix<float>(single_label, 1, 10);

	// train the neural network
	nn.train(training_set, labels, inputsNumber, 5000, 0.1f);

    // test on a single input after training
    output = nn.feedForward(single_input, 1, inputSize);
    cout << "Neural Network output for a single input after training:" << endl;
    printMatrix<float>(output.data(), 1, 10);
    cout << "Expected output:" << endl;
	printMatrix<float>(single_label, 1, 10);

	delete[] single_input;
	delete[] single_label;
    delete[] labels;
    delete[] training_set;
}




void tests() {
    // --- test neural network ---

    const unsigned int inputSize = 784;
    const unsigned int outputSize = 2;
    const unsigned int inputNumber = 2;


    float input[inputNumber * inputSize] = {
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

    float learningRate = 0.1f;
    float expected[inputNumber * 2] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    // real training
    // nn.setVerbose(true);
    nn.train(input, expected, inputNumber, 100, learningRate);
    cout << "Neural Network after training:" << endl << nn << endl;

    vector<float> finalAnswer = nn.feedForward(input, 2, inputSize);
    cout << "Neural Network output after training:" << endl;
    printMatrix<float>(finalAnswer.data(), inputNumber, outputSize);

    cout << endl;
}





int main()
{
    // --- Initialize GLFW + OpenGL ---
    init_gl();

    // main algorithm
    //tests();
    MNIST_neural_network_training();
	
	return 0;
}


// TODO : store all data in gl buffers to avoid cpu-gpu transfers
// TODO : coherence of height and width parameters in functions (sometimes swapped)







