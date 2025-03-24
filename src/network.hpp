#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

namespace BN {
class Network {

protected:
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

    std::vector<Layer> & layers() {
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

protected:
    inline static float rand_num(float min, float max) { // inclusive
        return (static_cast<float>(rand()) / RAND_MAX) * (max-min) + min;
    }
};
}