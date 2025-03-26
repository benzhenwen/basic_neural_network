#pragma once

#include <cmath>
#include <algorithm>
#include <vector>

#include <src/network.hpp>

namespace BN {
class NetworkBackpropagate {

protected:
    Network & n;

public:
    NetworkBackpropagate(Network & network):
    n(network) {}

    // TODO
    void train(std::vector<std::pair<std::vector<float>, std::vector<float>>> & data) {
        for (auto & set : data) train(set);
    }
    void train(std::pair<std::vector<float>, std::vector<float>> & data) {
        for (int i = 0; i < data.first.size(); i++) {
            n.input_layer()[i].raw_value = data.first[i];
        }
        n.calculate();
        backpropagate(data.second);
    }

    /*
    adjusts weights and biases based off of expected values
    you must set input_layer() and run calculate() first

    shoutout to 3blue1brown <3 https://youtu.be/tIeHLnjs5U8?si=2c2cXLjppweIrZRa
    */
    inline void backpropagate(std::vector<float> & expected_values) {
        std::vector<Network::Layer> & network = n.layers();

        auto expected_change = std::vector<float>(n.output_layer().size());

        // init expected change first as difference in output vs expected
        for (int i = 0; i < expected_change.size(); i++) {
            expected_change[i] = n.output_layer()[i].value() - expected_values[i];
        }

        // logger << expected_change[0];

        // main loop - from last layer to first (not including the input layer)
        for (int layer_i = network.size() - 1; layer_i > 0; layer_i--) {
            Network::Layer & layer = network[layer_i];
            auto prev_expected_change = std::vector<float>(network[layer_i-1].size());

            for (int node_i = 0; node_i < layer.size(); node_i++) {
                Network::Node & node = layer[node_i];
                const float d_error_d_value = Network::sigmoid_prime(node.raw()) * 2 * expected_change[node_i]; // (dC / dz(L) = s'(z(L)) * 2 * (a(L) - y) 

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
};
}