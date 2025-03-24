#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

namespace BN {
class Network {
friend class NetworkDrawer;
    static constexpr Logger logger = Logger("Network");

public:
    inline static float sigmoid(float input) {
        return input / (2*(1 + std::abs(input))) + 0.5;
    }
    inline static float sigmoid_prime(float input) {
        const float absp1 = std::abs(input) + 1;
        return 1 / (2 * (absp1 * absp1));
    }

public:
    struct Node {
        std::vector<float> weights;
        float bias;

        float raw_value;
        inline float value() {
            return sigmoid(raw_value);
        }
        inline float value_check_is_input(bool is_input_layer) {
            return is_input_layer ? raw() : value();
        }
        inline float raw() {
            return raw_value;
        }
    };

    struct Layer: public std::vector<Node> {
        bool is_input_layer;

        Layer(unsigned int size, unsigned int prev_layer_size): 
        std::vector<Node>(size),
        is_input_layer(false) {
            for (Node & n : *this) n.weights.resize(prev_layer_size);
        }
        Layer(unsigned int size): 
        std::vector<Node>(size),
        is_input_layer(true) {} 

        inline void calculate(Layer & input) {
            // reset 
            for (Node & n : *this) n.raw_value = 0;

            // apply weights
            for (int prev_i = 0; prev_i < input.size(); prev_i++) {
                for (int curr_i = 0; curr_i < size(); curr_i++) {
                    float prev_value = input[prev_i].value_check_is_input(input.is_input_layer);
                    at(curr_i).raw_value += at(curr_i).weights[prev_i] * prev_value; // summation of node weights times input values
                }
            }

            // apply bias
            for (int curr_i = 0; curr_i < size(); curr_i++) {
                at(curr_i).raw_value += at(curr_i).bias;
            }
        }
    };

protected:
    std::vector<Layer> network;

public:
    Network(unsigned int _input_layer_size) {
        network.push_back(Layer(_input_layer_size));
    }

    const std::vector<Layer> & access_network() {
        return network;
    }

    // access of input layer to modify values
    inline Layer & input_layer() {
        return network.front();
    }
    inline Layer & output_layer() {
        return network.back();
    }

    // append a new layer, or multiple layers with node count equal to size
    inline void append_layers(unsigned int size, unsigned int count = 1) {
        int prev_layer_size = network.back().size();
        for (long i = 0; i < count; i++) {
            network.push_back(Layer(size, prev_layer_size));
            prev_layer_size = size;
        }
    }

    /*
    randomly sets all the weights and biases of the network

    weight_variation - the variation of the weights. randomizes value from -x to x
    bias_variation - the variation of the bias. randomizes value from -x to x multiplied by the number of nodes in the previous layer
    */
    inline void randomize(float weight_variation = 1, float bias_variation = 0.5, long seed = std::time(0)) {
        srand(seed);

        unsigned int previous_layer_size = network[0].size();
        for (Layer & layer : network) {
            for (Node & node : layer) {
                std::generate(node.weights.begin(), node.weights.end(), [weight_variation]() { return rand_num(-weight_variation, weight_variation); });
                node.bias = rand_num(-bias_variation * previous_layer_size, bias_variation * previous_layer_size);
            }
            previous_layer_size = layer.size();
        }
    }

    /*
    updates all the values of the network.
    set the network's input layer first if needed.
    */
    inline void calculate() {
        for (int layer_i = 1; layer_i < network.size(); layer_i++) {
            network[layer_i].calculate(network[layer_i-1]);
        }
    }

    /*
    adjusts weights and biases based off of expected values
    you must set input_layer() and run calculate() first

    shoutout to 3blue1brown <3 https://youtu.be/tIeHLnjs5U8?si=2c2cXLjppweIrZRa
    */
    inline void backpropagate(std::vector<float> & expected_values) {
        auto expected_change = std::vector<float>(output_layer().size());

        // init expected change first as difference in output vs expected
        for (int i = 0; i < expected_change.size(); i++) {
            expected_change[i] = output_layer()[i].value() - expected_values[i];
        }

        // logger << expected_change[0];

        // main loop - from last layer to first (not including the input layer)
        for (int layer_i = network.size() - 1; layer_i > 0; layer_i--) {
            Layer & layer = network[layer_i];
            auto prev_expected_change = std::vector<float>(network[layer_i-1].size());

            for (int node_i = 0; node_i < layer.size(); node_i++) {
                Node & node = layer[node_i];
                const float d_error_d_value = sigmoid_prime(node.raw()) * 2 * expected_change[node_i]; // (dC / dz(L) = s'(z(L)) * 2 * (a(L) - y) 

                // weights
                for (int weight_i = 0; weight_i < node.weights.size(); weight_i++) {
                    node.weights[weight_i] += -1 * network[layer_i-1][weight_i].value_check_is_input(network[layer_i-1].is_input_layer) * d_error_d_value; // -(dz(L) / dw(L)) * (dC / dz(L)
                }

                // bias
                node.bias += -1 * d_error_d_value; // (dz(L) / db(L)) * (dC / dz(L))

                // expected change of prev layer
                for (int prev_i = 0; prev_i < network[layer_i-1].size(); prev_i++) {
                    prev_expected_change[prev_i] += node.weights[prev_i] * d_error_d_value; // (dz(L) / da(L-1)) * (dC / dz(L))
                }
            }

            expected_change = prev_expected_change;
        }
    }

protected:
    inline static float rand_num(float min, float max) { // inclusive
        return (static_cast<float>(rand()) / RAND_MAX) * (max-min) + min;
    }
};
}