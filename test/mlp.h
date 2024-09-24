#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace mlp_cpp {

template <typename T> std::vector<T> repeat_inner_vectors(const std::vector<std::vector<T>> &layer_outputs, int N) {
    // this function is to replicate how to stack over batch sizes in Swiftnet
    std::vector<T> result;

    // Iterate over each vector inside layer_outputs
    for (const auto &inner_vec : layer_outputs) {
        // For each element in the inner vector, repeat it N times
        for (int i = 0; i < N; ++i) {
            for (const T &element : inner_vec) {
                result.push_back(element);
            }
        }
    }

    return result;
}

template <typename SourceType, typename DestType>
std::vector<DestType> convert_vector(const std::vector<SourceType> &sourceVec) {
    std::vector<DestType> destVec;
    destVec.reserve(sourceVec.size());

    std::transform(sourceVec.begin(), sourceVec.end(), std::back_inserter(destVec),
                   [](const SourceType &val) { return static_cast<DestType>(val); });

    return destVec;
}

template <typename T> std::vector<T> stack_vector(const std::vector<T> &vec, size_t N) {
    // this function is to replicate how to stack over batch sizes in Swiftnet
    std::vector<T> result;
    result.reserve(vec.size() * N); // Reserve space to avoid reallocation
    for (size_t i = 0; i < N; ++i) {
        // Append a copy of the vector to the end of the result vector
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

// Define the Matrix struct with basic matrix-vector multiplication
template <typename T> struct Matrix {
    std::vector<std::vector<T>> data;

    // Default constructor that creates an empty matrix
    Matrix() {}
    // Constructor for the matrix of the given dimension with an initial value
    Matrix(std::size_t rows, std::size_t cols, T initialValue = T()) : data(rows, std::vector<T>(cols, initialValue)) {}

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (data.empty() || data[0].size() != vec.size()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        std::size_t rows = data.size();
        std::size_t cols = data[0].size();
        std::vector<T> result(rows, T());

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }

        return result;
    }

    // Transpose the matrix
    Matrix<T> transpose() const {
        // If the matrix is empty, return an empty matrix as the transpose
        if (data.empty()) return Matrix<T>();

        std::size_t rows = data.size();
        std::size_t cols = data[0].size();
        Matrix<T> transposed(cols, rows); // Note that the rows and cols are swapped

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                transposed.data[j][i] = data[i][j]; // Swap elements
            }
        }

        return transposed;
    }
    // Pretty print function for the matrix
    void print(int width = 10) const {
        for (const auto &row : data) {
            for (const T &val : row) {
                std::cout << std::setw(width) << val << " ";
            }
            std::cout << '\n';
        }
    }

    // Set linspace weights for the matrix
    void set_weights_linspace(T start, T end) {
        std::size_t total_elements = rows() * cols();
        T increment = (end - start) / (total_elements - 1);

        for (std::size_t i = 0; i < total_elements; ++i) {
            std::size_t row = i / cols();
            std::size_t col = i % cols();
            data[row][col] = start + i * increment;
        }
    }

    std::size_t rows() const { return data.size(); }
    std::size_t cols() const { return data.empty() ? 0 : data[0].size(); }
};

// Utility functions for linear and ReLU activations and their derivatives
double relu(double x) { return std::max(0.0, x); }

double linear(double x) {
    return x; // Identity function
}

double drelu(double x) { return x > 0 ? 1 : 0; }

double dlinear(double x) { return 1; }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double dsigmoid(double x) {
    // Calculating the derivative of the sigmoid function using its output
    // the input is already activated
    return x * (1 - x);
}
// MLP class using 'Matrix' struct for matrix operations
template <typename T> class MLP {
  private:
    std::vector<Matrix<T>> weights;
    int n_hidden_layers;
    int batch_size;

    int inputDim;
    int outputDim;
    int hiddenDim;

    std::string activation;
    std::string output_activation;
    // Utility function to initialize the weights randomly

    void initialize_random_weights(double weight_val_scaling_factor) {
        std::mt19937 gen(42);

        double xavier_stddev = std::sqrt(2.0 / (inputDim + outputDim));
        std::uniform_real_distribution<> dis(-weight_val_scaling_factor * xavier_stddev,
                                             weight_val_scaling_factor * xavier_stddev);

        for (auto &weight_matrix : weights) {
            for (auto &row : weight_matrix.data) {
                for (T &val : row) {
                    val = static_cast<T>(dis(gen));
                }
            }
        }
    }

  public:
    MLP(int inputDim_, int hiddenDim_, int outputDim_, int n_hidden_layers_, int batch_size_, std::string activation_,
        std::string output_activation_, std::string weight_init_mode)
        : n_hidden_layers(n_hidden_layers_), activation(activation_), output_activation(output_activation_),
          inputDim(inputDim_), outputDim(outputDim_), hiddenDim(hiddenDim_), batch_size(batch_size_) {
        /// TODO: normalisation of batch_size is only necessary because this is the implementation for batches of
        /// size 1. Later, make input tensor  not just a vector but a matrix (rows being batch_size dim). Then we
        /// don't need this

        // Initialize first layer weights
        weights.push_back(Matrix<T>(hiddenDim, hiddenDim)); // input_dim is padded by hiddenDim - inputDim

        // Initialize hidden layers weights
        for (int i = 0; i < n_hidden_layers - 2; ++i) {
            weights.push_back(Matrix<T>(hiddenDim, hiddenDim));
        }

        // Initialize output layer weights
        weights.push_back(Matrix<T>(hiddenDim, hiddenDim));

        // initialise weights
        double weight_val = 0.1;
        // linspace initialization of weights
        if (weight_init_mode == "linspace") {
            for (int i = 0; i < weights.size(); i++) {
                weights[i].set_weights_linspace(static_cast<T>(-weight_val * (i + 1)),
                                                static_cast<T>(weight_val * (i + 1)));
            }
        } else if (weight_init_mode == "random") {
            initialize_random_weights(1.0);
        } else {
            // default initialization of weights (to small random values or zeros)
            for (auto &weight_matrix : weights) {
                for (auto &row : weight_matrix.data) {
                    for (T &val : row) {
                        val = static_cast<T>(weight_val); // or any small number to initialize
                    }
                }
            }
        }

        // Pad the input matrix by setting cols after inputDim to zero
        for (std::size_t row = 0; row < hiddenDim; ++row) {
            for (std::size_t col = inputDim; col < hiddenDim; ++col) {
                weights.front().data[row][col] = (T)0.0f;
            }
        }

        // Pad the output matrix by setting rows after outputDim to zero:
        for (std::size_t row = outputDim; row < hiddenDim; ++row) {
            for (std::size_t col = 0; col < hiddenDim; ++col) {
                weights.back().data[row][col] = (T)0.0f;
            }
        }
    }
    std::vector<std::vector<T>> forward(const std::vector<T> &x, bool get_interm_fwd) {
        std::vector<std::vector<T>> layer_outputs(n_hidden_layers + 1);
        layer_outputs[0] = x;

        // Input to hidden layers
        for (int i = 0; i < n_hidden_layers - 1; i++) {
            // Multiply with weights (assuming matrix-vector multiplication)
            layer_outputs[i + 1] = weights[i] * layer_outputs[i];

            // Apply activation function for hidden layers
            for (T &val : layer_outputs[i + 1]) {
                if (activation == "relu") {
                    val = relu(val);
                } else if (activation == "sigmoid") {
                    val = sigmoid(val);
                } else {
                    val = linear(val);
                }
            }
        }

        // Hidden layer to output layer
        layer_outputs[n_hidden_layers] = weights[n_hidden_layers - 1] * layer_outputs[n_hidden_layers - 1];

        for (T &val : layer_outputs[n_hidden_layers]) {
            if (output_activation == "relu") {
                val = relu(val);
            } else if (output_activation == "sigmoid") {
                val = sigmoid(val);
            } else { // This covers "linear" or any unspecified activation, which defaults to linear
                val = linear(val);
            }
        }
        if (get_interm_fwd) {
            if (hiddenDim > layer_outputs[n_hidden_layers].size()) {
                // Calculate padding size
                std::size_t padding_size = hiddenDim - layer_outputs[n_hidden_layers].size();
                // Pad the last layer output with zeros
                layer_outputs[n_hidden_layers].insert(layer_outputs[n_hidden_layers].end(), padding_size, T(0));
                std::cout << "layer_outputs[n_hidden_layers] size " << layer_outputs[n_hidden_layers].size()
                          << std::endl;
            }
            return layer_outputs;
        } else {
            return {layer_outputs[n_hidden_layers]}; // Return final layer output
        }
    }

    // Implement the backward pass to compute gradients
    void backward(const std::vector<T> &input, const std::vector<T> &target, std::vector<Matrix<T>> &weight_gradients,
                  std::vector<std::vector<T>> &loss_grads, std::vector<T> &loss, std::vector<T> &dL_doutput,
                  std::vector<T> &dL_dinput, T loss_scale) {
        // Forward pass to get intermediate activations
        auto layer_outputs = forward(input, true);

        if (target.size() != layer_outputs.back().size()) {
            throw std::runtime_error(
                "Mismatch in sizes in mlp.h -> backward(). Target size: " + std::to_string(target.size()) +
                ", Layer output size: " + std::to_string(layer_outputs.back().size()) + ". They have to be the same.");
        }

        // Vectors to hold the gradients of the loss with respect to the activations
        std::vector<std::vector<T>> delta(layer_outputs.begin() + 1, layer_outputs.end());
        // Calculate the gradient for the output layer
        // Also, compute the MSE loss for the given batch
        for (std::size_t i = 0; i < delta.back().size(); ++i) {
            T error = (layer_outputs.back()[i] - target[i]);
            loss.push_back(error * error / (delta.back().size() * batch_size)); // Squared error for MSE
            delta.back()[i] = loss_scale * 2 * error / (delta.back().size());   // dLoss/dOutput;
        }

        dL_doutput = delta.back();
        for (int idx = 0; idx < delta.back().size(); idx++) {
            dL_doutput[idx] /= batch_size;
        }

        // calculate the interm_back (activated) backward pass for the last layer
        for (std::size_t i = 0; i < delta.back().size(); ++i) {
            if (output_activation == "relu") {
                delta.back()[i] *= drelu(layer_outputs.back()[i]); // ReLU derivative
            } else if (output_activation == "sigmoid") {
                delta.back()[i] *= dsigmoid(layer_outputs.back()[i]); // Sigmoid derivative
            } else {
                delta.back()[i] *= dlinear(layer_outputs.back()[i]); // Linear derivative
            }
        }

        // Go through layers in reverse order to propagate the error
        for (int i = n_hidden_layers - 2; i >= 0; --i) {
            // Calculate delta for next layer (i.e., previous in terms of forward pass)
            std::vector<T> new_delta(weights[i + 1].cols(), T(0));
            for (std::size_t col = 0; col < weights[i + 1].cols(); ++col) {
                for (std::size_t row = 0; row < weights[i + 1].rows(); ++row) {
                    new_delta[col] += delta[i + 1][row] * weights[i + 1].data[row][col];
                }
            }

            // Apply derivative of the activation function
            for (std::size_t j = 0; j < layer_outputs[i + 1].size(); ++j) {

                if (activation == "relu") {
                    new_delta[j] *= drelu(layer_outputs[i + 1][j]);
                } else if (activation == "sigmoid") {
                    new_delta[j] *= dsigmoid(layer_outputs[i + 1][j]);
                } else {
                    new_delta[j] *= dlinear(layer_outputs[i + 1][j]);
                }
            }
            delta[i] = new_delta;
        }

        for (int i = 0; i < delta.size(); i++) {
            auto loss_grad_el = delta[i];
            for (int idx = 0; idx < loss_grad_el.size(); idx++) {
                loss_grad_el[idx] /= batch_size;
            }
            loss_grads.push_back(loss_grad_el);
        }

        for (int i = n_hidden_layers - 1; i >= 0; i--) {

            // Initialize gradient matrix for next layer weights
            Matrix<T> layer_gradient(weights[i].rows(), weights[i].cols());
            // Gradient for this layer's weights
            for (std::size_t row = 0; row < layer_gradient.rows(); ++row) {
                for (std::size_t col = 0; col < layer_gradient.cols(); ++col) {
                    // note that layer_output[0] is input, thus this is the
                    // one from the previous layer
                    layer_gradient.data[row][col] = delta[i][col] * layer_outputs[i][row];
                }
            }
            weight_gradients[i] = layer_gradient.transpose();
            // std::cout << "Weight at " << i << std::endl;
            // weights[i].print();
            // std::cout << "Gradient of weight at " << i << std::endl;
            // weight_gradients[i].print();
        }

        // Calculate dL_dinput separately for clarity
        // Calculate delta for next layer (i.e., previous in terms of forward pass)
        std::vector<T> delta_input(weights[0].cols(), T(0));
        for (std::size_t col = 0; col < weights[0].cols(); ++col) {
            for (std::size_t row = 0; row < weights[0].rows(); ++row) {
                delta_input[col] += delta[0][row] * weights[0].data[row][col];
            }
        }

        // Apply derivative of the activation function, dL_dinput is unactivated (linear)
        for (std::size_t j = 0; j < layer_outputs[0].size(); ++j) {
            delta_input[j] *= dlinear(layer_outputs[0][j]) / batch_size;
        }
        dL_dinput = delta_input;
    }

    std::vector<T> getUnpackedWeights() const {
        std::vector<T> all_weights;
        for (const Matrix<T> &weight_matrix : weights) {
            for (const std::vector<T> &row : weight_matrix.transpose().data) {
                // using transpose as tiny-nn expects (input_width, width) as input matrix whereas we define it as
                // (width, input_width here)
                all_weights.insert(all_weights.end(), row.begin(), row.end());
            }
        }

        return all_weights;
    }
};
} // namespace mlp_cpp