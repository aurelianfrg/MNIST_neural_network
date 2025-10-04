// MNIST_neural_network.cpp : définit le point d'entrée de l'application.


#include "mnist_reader_less.hpp"
#include "gpu_matrix_op.hpp"
#include "neural_network.hpp"
#include "MNIST_neural_network.hpp"

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

    // Setup MNIST dataset
    mnist::MNIST_dataset<uint8_t, uint8_t> reader = mnist::read_dataset<uint8_t, uint8_t>();

    const unsigned int inputSize = 28 * 28; // size of input layer = number of pixels in an image
	const unsigned int outputSize = 10;     // size of output layer = number of possible digits
    const unsigned int trainingInputsNumber = 1000;
	const unsigned int testInputsNumber = 100;
	const float learningRate = 0.2f;
	const unsigned int epochs = 800;


	// to be used as input to the neural network, the rows must represent the pixels of an image and the columns the different images
	// using row-major storage, we need to transpose the dataset
    // this means that in memory, all the first pixels of each image will be stored in a row, then the second pixels ...

    float* training_set = setup_dataset(reader.training_images, inputSize, trainingInputsNumber);
    float* training_labels = setup_labels(reader.training_labels, trainingInputsNumber);
	float* test_set = setup_dataset(reader.test_images, inputSize, testInputsNumber);
	float* test_labels = setup_labels(reader.test_labels, testInputsNumber);

    // setup the neural network    
    vector<unsigned int> neuronsPerLayer({ 784, 784, 50, outputSize });
    NeuralNetwork<float> nn(4, neuronsPerLayer, inputSize);

	// train the neural network
	nn.train(training_set, training_labels, trainingInputsNumber, epochs, learningRate);

    // test after training
    vector<float> output = nn.feedForward(training_set, trainingInputsNumber, inputSize);
    float rate = conformRate(output.data(), training_labels, trainingInputsNumber, outputSize);
    cout << "Neural Network conform rate on training set after training (" << trainingInputsNumber << " inputs) : " << fixed << setprecision(2) << rate * 100 << "%" << endl;

	output = nn.feedForward(test_set, testInputsNumber, inputSize);
	rate = conformRate(output.data(), test_labels, testInputsNumber, outputSize);
    cout << "Neural Network conform rate on test set after training (" << testInputsNumber << " inputs) : " << fixed << setprecision(2) << rate * 100 << "%" << endl;

    //visualisation
    printMnistDigit(reader.test_images[0].data(), 28);
    cout << "Label: " << int(reader.test_labels[0]) << endl;
	float* single_input = setup_dataset(reader.test_images, inputSize, 1);
    vector<float> single_output = nn.feedForward(single_input, 1, inputSize);
    cout << "Neural Network output for this image :" << endl;
    printMatrix<float>(single_output.data(), 1, outputSize);

	// cleanup
    delete[] single_input;
	delete[] test_set;
	delete[] test_labels;
    delete[] training_labels;
    delete[] training_set;
}


float conformRate(const float* output, const float* expected, const unsigned int inputsNumber, const unsigned int outputSize) {
    int conformCount = 0;
    for (int input_rank = 0; input_rank < inputsNumber; ++input_rank) {
        // find the index of the maximum value in output and expected
        int maxOutputIndex = 0;
        int maxExpectedIndex = 0;
        float maxOutputValue = output[input_rank];
        float maxExpectedValue = expected[input_rank];
        for (int i = 1; i < outputSize; ++i) {
            if (output[input_rank + i * inputsNumber] > maxOutputValue) {
                maxOutputValue = output[input_rank + i * inputsNumber];
                maxOutputIndex = i;
            }
            if (expected[input_rank + i * inputsNumber] > maxExpectedValue) {
                maxExpectedValue = expected[input_rank + i * inputsNumber];
                maxExpectedIndex = i;
            }
        }
        if (maxOutputIndex == maxExpectedIndex) {
            conformCount++;
        }
    }
    return float(conformCount) / inputsNumber;
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
// TODO : coherence of height and width parameters in functions (sometimes swapped) : standard is height first then width







