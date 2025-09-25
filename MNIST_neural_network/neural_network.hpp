#pragma once

#include <iostream>
#include <vector>
#include <algorithm>

#include "gpu_matrix_op.hpp"

using namespace std;

template <typename parameters_t>
class NeuralNetwork {
	// represents a neural network constituted of layers

	friend ostream& operator << <>(ostream& os, const NeuralNetwork<parameters_t>& nn);

protected:
	unsigned int layersNumber;					//nb of layers
	vector<parameters_t*> layersWeights;		//vector of layersNumber pointers to layers weights, contiguously allocated 2D arrays of weights 
	vector<parameters_t*> layersBias;
	vector<int> widths;							//widths of each layer
	vector<int> heights;						//heights of each layer
	vector<parameters_t*> cachedLayersOutputs;	//cached outputs of each layer to be used during backpropagation				

public:
	NeuralNetwork(unsigned int layersNumber, vector<int> widths, vector<int> heights) : 
		layersNumber(layersNumber), 
		widths(widths), 
		heights(heights) 
	{

		if (layersNumber != widths.size() or layersNumber != heights.size()) {
			throw invalid_argument("layersNumber must be equal to the size of widths and heights vectors");
		}
		layersBias.resize(layersNumber);
		layersWeights.resize(layersNumber);
		cachedLayersOutputs.resize(layersNumber);

		//allocating weights layers and bias layers with corresponding sizes
		for (int i = 0; i < layersNumber; ++i) {
			layersWeights.at(i) = new parameters_t[widths.at(i) * heights.at(i)]; // one weight per connection
		}
		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			layersBias.at(i) = new parameters_t[heights.at(i)]; // one bias per neuron
		}
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
	}

	void random_init(int seed) {
		//initialize weights and bias with random values between 0 and 1

		srand(seed+1);
		
		for (int i = 0; i < layersNumber; ++i) { // fill each layer
			for (int neuron_rank = 0; neuron_rank < widths.at(i) * heights.at(i); ++neuron_rank) {
				parameters_t rand_value = 2.0 * (parameters_t(rand()) / RAND_MAX) - 1.0;
				layersWeights[i][neuron_rank] = rand_value;
			}
		}

		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			for (int neuron_rank = 0; neuron_rank < heights.at(i); ++neuron_rank) {
				parameters_t rand_value = parameters_t(rand()) / RAND_MAX;
				layersBias[i][neuron_rank] = rand_value;
			}
		}
	}

	vector<parameters_t> feedForward(
		parameters_t* input,	// input matrix containing one column per input
		unsigned int width,		// number of inputs (number of columns of the input matrix)
		unsigned int height)	// size of each input (number of rows of the input matrix)
	{
		// feed forward the input through the neural network and return the output of the last layer

		// input dimension must match the input layer dimension
		if ( height != widths.at(0)) {
			throw invalid_argument("incorrect input dimensions");
		}

		GLuint ssboWeighted, ssboBiased, ssboActivated;
		glGenBuffers(1, &ssboWeighted);
		glGenBuffers(1, &ssboActivated);
		glGenBuffers(1, &ssboBiased);

		// apply weights to input
		matrix_mult<parameters_t>(layersWeights[0], input, ssboWeighted, heights.at(0), widths.at(0), width);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeighted);
		parameters_t* weighted = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

		cout << "After weights multiplication:" << endl;
		printMatrix<parameters_t>(weighted, width, height);

		//apply activation to input
		sigmoid_activation<parameters_t>(weighted, ssboActivated, height, width);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboActivated);
		parameters_t* activated = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

		cout << "After activation:" << endl;
		printMatrix<parameters_t>(activated, width, height);

		// feed forward through the layers
		for (int step = 1; step < layersNumber; ++step) {
			// apply weights to input
			matrix_mult<parameters_t>(layersWeights[step], input, ssboWeighted, heights.at(step), widths.at(step), width);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboWeighted);
			parameters_t* weighted = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

			cout << "After weights multiplication:" << endl;
			printMatrix<parameters_t>(weighted, width, heights.at(step));

			// add bias
			matrix_add_constant_vec<parameters_t>(weighted, layersBias[step], ssboBiased, width, heights.at(step));
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBiased);
			parameters_t* biased = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

			cout << "After bias addition:" << endl;
			printMatrix<parameters_t>(biased, width, heights.at(step));

			//apply activation to input
			sigmoid_activation<parameters_t>(weighted, ssboActivated, height, width);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboActivated);
			parameters_t* activated = (parameters_t*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

			cout << "After activation:" << endl;
			printMatrix<parameters_t>(activated, width, heights.at(step));
		}

		// there are "width" outputs (= as many as inputs) and they are of the same size as the number of neurons in the last layer
		vector<parameters_t> output(width * heights.back());
		for (int i = 0; i < width * heights.back(); ++i) {
			output[i] = activated[i];
		}

		glDeleteBuffers(1, &ssboWeighted);
		glDeleteBuffers(1, &ssboActivated);
		glDeleteBuffers(1, &ssboBiased);
		return output;
	}
};

template <typename parameters_t>
ostream & operator <<(ostream &os, const NeuralNetwork<parameters_t> & nn) {
	//print the neural network weights and bias
	for (int i = 0; i < nn.layersNumber; ++i) {
		os << "Layer " << i << " weights:" << endl;
		for (int j = 0; j < nn.heights[i]; ++j) {
			for (int k = 0; k < nn.widths[i]; ++k) {
				// printing one less digit if negative
				if (nn.layersWeights.at(i)[j * nn.widths[i] + k] < 0) {
					os << std::fixed << std::setprecision(1) << nn.layersWeights.at(i)[j * nn.widths[i] + k] << " ";
				}
				else {
					os << std::fixed << std::setprecision(2) << nn.layersWeights.at(i)[j * nn.widths[i] + k] << " ";
				}
			}
			os << endl;
		}
		os << endl;
	}
	for (int i = 1; i < nn.layersNumber; ++i) { // input layer has no bias
		os << "Layer " << i << " bias:" << endl;
		for(int j = 0; j < nn.heights[i]; ++j) {
			os << std::fixed << std::setprecision(2) << nn.layersBias.at(i)[j] << endl;
		}
		os << endl;
	}
	return os;
}