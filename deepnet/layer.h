#pragma once

#include "activation_function.h"
#include <random>

class Layer {
protected:
	int neuron, input_size = 0;

public:
	Layer(int input_size, int neuron) : input_size(input_size), neuron(neuron) {}

	void set_input_size(int input_size) {
		this->input_size = input_size;
	}

	virtual void initialize_weight(int input_size) = 0;

	int get_input_size() {
		return input_size;
	}

	int get_output_size() {
		return neuron;
	}

	virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input ) = 0;

	virtual Eigen::MatrixXd back_propagation(const Eigen::MatrixXd& output_gradient, double learning_rate) = 0;

	virtual void update_weight(Eigen::MatrixXd weight_gradient, Eigen::VectorXd bias_gradient, double learning_rate) = 0;
};

class Dense : public Layer {
private:
	std::unique_ptr<Activation> activation;
	Eigen::MatrixXd weight, intermediate;
	Eigen::VectorXd bias;

public:

	//for first layer
	Dense(std::string activation_function, int input_size, int neuron) : Layer(input_size, neuron){
		this->initialize_weight(input_size);
		bias = Eigen::VectorXd::Zero(neuron); // Initialize bias
		activation = assign_activation(activation_function); // Initialize activation
	}

	Dense(std::string activation_function, int neuron) : Layer(0, neuron) {
		bias = Eigen::VectorXd::Zero(neuron); // Initialize bias
		activation = assign_activation(activation_function); // Initialize activation
	}

	void initialize_weight(int input_size) override {
		double stddev = std::sqrt(2.0 / input_size); // Standard deviation for He initialization
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<> dist(0.0, stddev);
		// Initialize weights with He Normal
		weight = Eigen::MatrixXd::Zero(input_size, neuron).unaryExpr([&](double) { return dist(gen); });
	}

	Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override {
		Eigen::MatrixXd output = input * weight;
		output.rowwise() += bias.transpose();
		intermediate = input;

		return activation->activate(output);
	}

	Eigen::MatrixXd back_propagation(const Eigen::MatrixXd& output_gradient, double learning_rate) override {
		Eigen::MatrixXd activation_gradient = activation->derivative(intermediate * weight);
		// Since output_gradient = output - target, no need to apply softmax derivative
		Eigen::MatrixXd grad_pre_activation;
		if (typeid(*activation) == typeid(Softmax)) grad_pre_activation = output_gradient;
		else grad_pre_activation = output_gradient.cwiseProduct(activation_gradient);

		Eigen::MatrixXd weight_gradient = intermediate.transpose() * grad_pre_activation;
		Eigen::VectorXd bias_gradient = grad_pre_activation.colwise().sum();

		this->update_weight(weight_gradient, bias_gradient, learning_rate);

		// Compute the gradient to propagate to the previous layer (input gradient)
		Eigen::MatrixXd input_gradient = grad_pre_activation * weight.transpose();
		return input_gradient;
	}


	void update_weight(Eigen::MatrixXd weight_gradient, Eigen::VectorXd bias_gradient, double learning_rate) override {
		weight -= learning_rate * weight_gradient;
		bias -= learning_rate * bias_gradient;
	}

};