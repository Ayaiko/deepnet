#pragma once

#include <vector>
#include <memory>
#include "layer.h"
#include "loss_function.h"
#include "optimizer.h"

class Model {
private:
	std::vector<std::unique_ptr<Layer>> layers;
	std::unique_ptr<Optimizer> opt;
	std::unique_ptr<Loss_function> loss;
	int batch_size = 32;//set
	double learning_rate = 0.01;// set 

public:
	Model() {
		loss = std::make_unique<Categorical_cross_entropy>();
		//opt = std::make_unique<>();
	}

	void add_layer(std::unique_ptr<Layer> layer) {
		if(layers.empty() && layer->get_input_size() == 0) 
			throw std::runtime_error("The input size of the first layer is required");

		if (!layers.empty()) {
			int prev_output_size = layers.back()->get_output_size();
			layer->set_input_size(prev_output_size);
			layer->initialize_weight(prev_output_size);
		}

		layers.push_back(std::move(layer));
	}

	void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
		int num_samples = x.rows(); 

		// Iterate over the dataset in batches
		for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
			// Calculate the end index of the current batch
			int batch_end = std::min(batch_start + batch_size, num_samples);

			// Extract the current batch of inputs and targets
			Eigen::MatrixXd batch_x = x.middleRows(batch_start, batch_end - batch_start); // Shape: (batch_size, num_features)
			Eigen::MatrixXd batch_y = y.middleRows(batch_start, batch_end - batch_start); // Shape: (batch_size, num_targets)

			// Forward pass through all layers
			Eigen::MatrixXd output = batch_x;
			for (int i = 0; i < layers.size(); i++) {
				output = layers[i]->forward(output);
			}

			// Compute the loss gradient for the current batch
			Eigen::MatrixXd gradient = loss->derivative(output, batch_y);

			// Backward pass through all layers
			for (int i = layers.size() - 1; i >= 0; i--) {
				gradient = layers[i]->back_propagation(gradient, learning_rate);
			}

		}


	}

	void predict() {



	}

	void compile() {
		/*
		set loss function
		set optimizer
		set batch size
		*/
	}
};