#pragma once

#include <Eigen/Core>
#include <memory>
#include <cmath>

class Activation;


class Activation {
public:
	virtual Eigen::MatrixXd activate(const Eigen::MatrixXd& input) = 0;
	virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) = 0;
};

class Relu : public Activation {
public:
	Eigen::MatrixXd activate(const Eigen::MatrixXd& input) override {
		return input.cwiseMax(0.0);
	}

	Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) override {
		Eigen::MatrixXd grad = (input.array() > 0).cast<double>();
		return grad;
	}
};

class Softmax : public Activation {
public:
	Eigen::MatrixXd activate(const Eigen::MatrixXd& input) override {
		Eigen::MatrixXd stabilized_input = input.colwise() - input.rowwise().maxCoeff();
		Eigen::MatrixXd exp_input = stabilized_input.array().exp();
		Eigen::VectorXd row_sum = exp_input.rowwise().sum();
		Eigen::MatrixXd softmax = exp_input.array().colwise() / row_sum.array();
		//std::cout << softmax << std::endl;
		return softmax;
	}

	Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) override {
		Eigen::MatrixXd softmax_output = this->activate(input); // Get softmax output

		int n = softmax_output.rows(); // Number of samples (batch size)
		int m = softmax_output.cols(); // Number of classes

		Eigen::MatrixXd jacobian_matrix(n, m * m); // Jacobian matrix of size (n x m^2)

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				for (int k = 0; k < m; ++k) {
					if (j == k) {
						jacobian_matrix(i, j * m + k) = softmax_output(i, j) * (1 - softmax_output(i, j)); // Diagonal terms
					}
					else {
						jacobian_matrix(i, j * m + k) = -softmax_output(i, j) * softmax_output(i, k); // Off-diagonal terms
					}
				}
			}
		}

		return jacobian_matrix;
	}
};

std::unique_ptr<Activation> assign_activation(std::string func) {
	if (func == "Relu" || func == "relu") return std::make_unique<Relu>();
	else if (func == "Softmax" || func == "softmax") return std::make_unique<Softmax>();

	throw std::invalid_argument("Unknown activation function: " + func);
}
