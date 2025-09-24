#pragma once

#include <iostream>
#include <vector>
#include <algorithm>

#include "gpu_matrix_op.hpp"

using namespace std;

template <int layersNumber, typename parameters_t>
class NeuralNetwork {
	// represents a neural network constituted of layers
protected:
	vector<parameters_t*> layersWeights; //vector of layersNumber pointers to layers weights, contiguously allocated 2D arrays of weights 
	vector<parameters_t*> layersBias;
	int widths[layersNumber]; //widths of each layer
	int heights[layersNumber]; //heights of each layer

public:
	NeuralNetwork(int widths[layersNumber], int heights[layersNumber]) {
		copy(widths, widths + layersNumber, this->widths);
		copy(heights, heights + layersNumber, this->heights);
		layersBias.resize(layersNumber);
		layersWeights.resize(layersNumber);

		//allocating weights layers and bias layers with corresponding sizes
		for (int i = 0; i < layersNumber; ++i) {
			layersWeights[i] = new parameters_t[widths[i] * heights[i]];
		}
		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			layersBias[i] = new parameters_t[widths[i] * heights[i]];
		}
		this->random_init(1729);
	}

	~NeuralNetwork() {
		//deallocating weights layers and bias layers
		for (int i = 0; i < layersNumber; ++i) {
			delete[] layersWeights[i];
		}
		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			delete[] layersBias[i];
		}
	}

	void random_init(int seed) {
		//initialize weights and bias with random values between 0 and 1

		srand(seed);
		
		for (int i = 0; i < layersNumber; ++i) { // fill each layer
			for (int neuron_rank = 0; neuron_rank < widths[i] * heights[i]; ++neuron_rank) {
				parameters_t rand_value = parameters_t(rand()) / RAND_MAX;
				layersWeights[i][neuron_rank] = rand_value;
			}
		}

		for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
			for (int neuron_rank = 0; neuron_rank < widths[i] * heights[i]; ++neuron_rank) {
				parameters_t rand_value = parameters_t(rand()) / RAND_MAX;
				layersBias[i][neuron_rank] = rand_value;
			}
		}
	}

	friend ostream & operator << <>(ostream& os, const NeuralNetwork<layersNumber, parameters_t> & nn);

};

template <int layersNumber, typename parameters_t>
ostream & operator <<(ostream &os, const NeuralNetwork<layersNumber, parameters_t> & nn) {
	//print the neural network weights and bias
	for (int i = 0; i < layersNumber; ++i) {
		os << "Layer " << i << " weights:" << endl;
		for (int j = 0; j < nn.heights[i]; ++j) {
			for (int k = 0; k < nn.widths[i]; ++k) {
				os << std::fixed << std::setprecision(2) << nn.layersWeights[i][j * nn.widths[i] + k] << " ";
			}
			os << endl;
		}
		os << endl;
	}
	for (int i = 1; i < layersNumber; ++i) { // input layer has no bias
		os << "Layer " << i << " bias:" << endl;
		for(int j = 0; j < nn.heights[i]; ++j) {
			for (int k = 0; k < nn.widths[i]; ++k) {
				os << std::fixed << std::setprecision(2) << nn.layersBias[i][j * nn.widths[i] + k] << " ";
			}
			os << endl;
		}
		os << endl;
	}
	return os;
}