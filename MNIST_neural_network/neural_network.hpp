#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

#include "gpu_matrix_op.hpp"

using namespace std;

template <typename parameters_t>
class NeuralNetwork {
	// represents a neural network constituted of layers

	template <typename parameters_t>
	friend ostream& operator <<(ostream& os, const NeuralNetwork<parameters_t>& nn);

protected:
	unsigned int layersNumber;					//nb of layers
	vector<parameters_t*> layersWeights;		//vector of layersNumber pointers to layers weights, contiguously allocated 2D arrays of weights 
	vector<parameters_t*> layersBias;
	vector<unsigned int> neuronsPerLayer;		//number of neurons per layer
	unsigned int inputSize;						//size of the input layer (number of inputs)

	vector<GLuint> cachedLayersOutputsSsbos;	//cached outputs of each layer (state of every neuron) to be used during backpropagation



public:
	NeuralNetwork(unsigned int layersNumber, vector<unsigned int> neurons, unsigned int inputSize) : 
		layersNumber(layersNumber), 
		neuronsPerLayer(neurons),
		inputSize(inputSize)
	{

		if (layersNumber != neuronsPerLayer.size() ) {
			throw invalid_argument("layersNumber must be equal to the size of widths and heights vectors");
		}
		layersBias.resize(layersNumber);
		layersWeights.resize(layersNumber);
		cachedLayersOutputsSsbos.resize(layersNumber);

		//allocating weights layers and bias layers with corresponding sizes
		layersWeights.at(0) = new parameters_t[inputSize * neurons.at(0)]; // one weight per connection between input and first layer
		for (int i = 1; i < layersNumber; ++i) {
			layersWeights.at(i) = new parameters_t[neurons.at(i-1) * neurons.at(i)]; // one weight per connection between neurons of layer l-1 and l
		}
		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			layersBias.at(i) = new parameters_t[neurons.at(i)]; // one bias per neuron
		}

		//create as many ssbo as layers to cache the outputs of each layer
		for (int i = 0; i < layersNumber; ++i) {
			GLuint ssbo;
			glGenBuffers(1, &ssbo);
			cachedLayersOutputsSsbos.at(i) = ssbo;
		}

		// initializing weights and bias with random values
		this->random_init(1729);
	}

	~NeuralNetwork() {
		//deallocating weights layers and bias layers
		for (int i = 0; i < layersNumber; ++i) {
			delete[] layersWeights.at(i);
		}
		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			delete[] layersBias.at(i);
		}
		//delete ssbo used to cache layers outputs
		for (int i = 0; i < layersNumber; ++i) {
			glDeleteBuffers(1, &cachedLayersOutputsSsbos.at(i));
		}
	}

	void random_init(int seed) {
		//initialize weights and bias with random values between 0 and 1

		srand(seed);
		
		for (int neuron_rank = 0; neuron_rank < inputSize * neuronsPerLayer.at(0); ++neuron_rank) {
			parameters_t rand_value = 2.0 * (parameters_t(rand()) / RAND_MAX) - 1.0;
			layersWeights[0][neuron_rank] = rand_value;
		}
		for (int i = 1; i < layersNumber; ++i) { // fill each layer
			for (int neuron_rank = 0; neuron_rank < neuronsPerLayer.at(i-1) * neuronsPerLayer.at(i); ++neuron_rank) {
				parameters_t rand_value = 2.0 * (parameters_t(rand()) / RAND_MAX) - 1.0;
				layersWeights[i][neuron_rank] = rand_value;
			}
		}

		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			for (int neuron_rank = 0; neuron_rank < neuronsPerLayer.at(i); ++neuron_rank) {
				parameters_t rand_value = parameters_t(rand()) / RAND_MAX;
				layersBias[i][neuron_rank] = rand_value;
			}
		}
	}

	vector<parameters_t> feedForward(
		parameters_t* input,	// input matrix containing one column per input
		unsigned int sampleSize,		// number of inputs (number of columns of the input matrix)
		unsigned int vectorSize)	// size of each input (number of rows of the input matrix)
	{
		// feed forward the input through the neural network and return the output of the last layer

		// input dimension must match the input layer dimension
		if ( vectorSize != inputSize) {
			throw invalid_argument("incorrect input dimensions");
		}

		GLuint ssboWeighted, ssboBiased;
		glGenBuffers(1, &ssboWeighted);
		glGenBuffers(1, &ssboBiased);
		parameters_t *activated, *weighted, *biased;

		// apply weights to input
		matrix_mult<parameters_t>(layersWeights[0], input, ssboWeighted, inputSize, neuronsPerLayer.at(0), sampleSize);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeighted);
		weighted = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

		cout << "After weights multiplication:" << endl;
		printMatrix<parameters_t>(weighted, sampleSize, neuronsPerLayer.at(0));

		//apply activation to input
		sigmoid_activation<parameters_t>(weighted, cachedLayersOutputsSsbos.at(0), neuronsPerLayer.at(0), sampleSize);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, cachedLayersOutputsSsbos.at(0));
		activated = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

		cout << "After activation:" << endl;
		printMatrix<parameters_t>(activated, sampleSize, neuronsPerLayer.at(0));

		// feed forward through the layers
		for (int layer = 1; layer < layersNumber; ++layer) {
			// apply weights to input
			matrix_mult<parameters_t>(layersWeights[layer], activated, ssboWeighted, neuronsPerLayer.at(layer), neuronsPerLayer.at(layer-1), sampleSize);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeighted);
			weighted = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

			cout << "After weights multiplication:" << endl;
			printMatrix<parameters_t>(weighted, sampleSize, neuronsPerLayer.at(layer));

			// add bias
			matrix_add_constant_vec<parameters_t>(weighted, layersBias[layer], ssboBiased, sampleSize, neuronsPerLayer.at(layer));
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBiased);
			biased = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

			cout << "After bias addition:" << endl;
			printMatrix<parameters_t>(biased, sampleSize, neuronsPerLayer.at(layer));

			//apply activation to input
			sigmoid_activation<parameters_t>(weighted, cachedLayersOutputsSsbos.at(layer), neuronsPerLayer.at(layer), sampleSize);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, cachedLayersOutputsSsbos.at(layer));
			activated = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

			cout << "After activation:" << endl;
			printMatrix<parameters_t>(activated, sampleSize, neuronsPerLayer.at(layer));
		}

		// there are "sampleSize" outputs (= as many as inputs) and they are of the same size as the number of neurons in the last layer
		vector<parameters_t> output(sampleSize * neuronsPerLayer.back());
		for (int i = 0; i < sampleSize * neuronsPerLayer.back(); ++i) {
			output[i] = activated[i];
		}

		glDeleteBuffers(1, &ssboWeighted);
		glDeleteBuffers(1, &ssboBiased);
		return output;
	}

	parameters_t loss(parameters_t* predicted, parameters_t* expected) {
		// binary cross-entropy loss function
		// input vectors are expected to be of the same size as the output layer
		int vectorSize = neuronsPerLayer.back();
		parameters_t totalLoss = 0.0;

		for (int i = 0; i < vectorSize; ++i) {
			parameters_t singleOutputLoss = -(expected[i] * log(predicted[i]) + (1 - expected[i]) * log(1 - predicted[i]));
			totalLoss += singleOutputLoss;
		}
		return totalLoss / vectorSize; 
	}

	parameters_t cost(parameters_t* predicted, parameters_t* expected, unsigned int inputsNumber) {
		int vectorSize = neuronsPerLayer.back();
		parameters_t totalCost = 0.0;	

		for (int input_rank = 0; input_rank < inputsNumber; ++input_rank) {
			parameters_t* predicted_vector = &predicted[input_rank * vectorSize];
			parameters_t* expected_vector = &expected[input_rank * vectorSize];
			totalCost += loss(predicted_vector, expected_vector);
		}
		return totalCost / inputsNumber; // mean cost over all inputs
	}

	void backPropagation(parameters_t* expected) {

		// --- Compute cost for the last prediction ---
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, cachedLayersOutputsSsbos.back());
		parameters_t* A_L = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		int outputSize = neuronsPerLayer.back();


		// --- Cost gradient computation ---

		// calculate dC_dZ(L) for the output layer

		// calculate dC_dW(L) and dC_db(L) for the output layer

		// calculate dC_dA(L-1) to be used for the previous layer

		// for each previous layer l = L-1 ... 1
			// calculate dC_dZ(l)
			// calculate dC_dW(l) and dC_db(l)
			// calculate dC_dA(l-1) to be used for the previous layer

	}
};

template <typename parameters_t>
ostream & operator <<(ostream &os, const NeuralNetwork<parameters_t> & nn) {
	//print the neural network weights and bias

	os << "Layer " << 0 << " weights:" << endl;
	printMatrix<parameters_t>(nn.layersWeights.at(0), nn.inputSize, nn.neuronsPerLayer.at(0));
	for (int i = 1; i < nn.layersNumber; ++i) {
		os << "Layer " << i << " weights:" << endl;
		printMatrix<parameters_t>(nn.layersWeights.at(i), nn.neuronsPerLayer.at(i-1), nn.neuronsPerLayer.at(i));
	}

	for (int i = 1; i < nn.layersNumber; ++i) { // input layer has no bias
		os << "Layer " << i << " bias:" << endl;
		for(int j = 0; j < nn.neuronsPerLayer.at(i); ++j) {
			os << std::fixed << std::setprecision(2) << nn.layersBias.at(i)[j] << endl;
		}
		os << endl;
	}
	return os;
}