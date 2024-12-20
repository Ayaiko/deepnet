#pragma once

class Loss_function {
public:
	//need cross entropy loss
	virtual Eigen::MatrixXd activate(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target) = 0;
	virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target) = 0;
};

class Categorical_cross_entropy : public Loss_function {
public:
    Eigen::MatrixXd activate(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target) override {
        int batch_size = output.rows();
        Eigen::MatrixXd loss(batch_size, 1);

        for (int i = 0; i < batch_size; ++i) {
            double log_sum = 0.0;
            for (int j = 0; j < output.cols(); ++j) {
                if (target(i, j) == 1) {
                    log_sum -= target(i, j) * log(output(i, j));
                }
            }
            loss(i, 0) = log_sum;
        }
        return loss;
    }

    Eigen::MatrixXd derivative(const Eigen::MatrixXd& output, const Eigen::MatrixXd& target) override {
        return output - target;  
    }
};