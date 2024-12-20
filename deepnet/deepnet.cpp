#include <iostream>
#include <Eigen/Dense>
#include "model.h"
#include <vector>
#include "mnist_reader_less.hpp"

using Eigen::MatrixXd;
using namespace std;
 
int main()
{
	Model model;

	auto dataset = mnist::read_dataset<std::uint8_t, std::uint8_t>();

	model.add_layer(make_unique<Dense>("relu", 784, 5));
	model.add_layer(make_unique<Dense>("relu", 7));
	model.add_layer(make_unique<Dense>("softmax", 10));

	std::vector<uint8_t> image_data = dataset.training_images[0];  // A flat vector representing one image
	std::vector<uint8_t> target_data = dataset.training_labels;

	int size = 200;
	Eigen::MatrixXd train(size, 784), target(size, 10);  // Create a vector of size 784
	target.setZero();

	for (int i = 0; i < size; i++) {
		target(i, static_cast<int>(target_data[i])) = 1;
		for (int j = 0; j < 784; j++) {
			train(i, j) = static_cast<double>(dataset.training_images[i][j]);  // Copy pixel data into the vector
		}
	}
	
	// Now you can pass the image_matrix to your train function
	model.train(train, target);
	return 0;
}