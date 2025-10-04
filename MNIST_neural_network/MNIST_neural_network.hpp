// MNIST_neural_network.cpp : définit le point d'entrée de l'application.


#include "mnist_reader_less.hpp"
#include "gpu_matrix_op.hpp"
#include "neural_network.hpp"

using namespace std;

void printMnistDigit(uint8_t* img, int size);

float* setup_dataset(vector<vector<uint8_t>>& raw_set, const unsigned int inputSize, const unsigned int inputsNumber);
    // to be used as input to the neural network, the rows must represent the pixels of an image and the columns the different images
    // using row-major storage, we need to transpose the dataset
    // this means that in memory, all the first pixels of each image will be stored in a row, then the second pixels ...

float* setup_labels(vector<uint8_t> raw_labels, const unsigned int inputsNumber);
    // to be used to train the neural network, labels must be presented as bitmaps of size 10 with a single bit being at 1 : the bit of rank equal to the digit value

void MNIST_neural_network_training();

float conformRate(const float* output, const float* expected, const unsigned int inputsNumber, const unsigned int outputSize);

void tests();


// TODO : store all data in gl buffers to avoid cpu-gpu transfers
// TODO : coherence of height and width parameters in functions (sometimes swapped)







