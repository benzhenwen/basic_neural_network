#pragma once

#include <vector>
#include <string>

#include <glm/glm.hpp>

#include <GLFWE/text/character_set.hpp>
#include <GLFWE/shape/quad.hpp>

#include <src/network.hpp>


namespace BN {
class NetworkDrawer {

protected:
    NetworkDrawer() = delete;

public:
    static glm::vec3 node_color(float value) {
        float g = (value + 1) / 2;
        return glm::vec3(g, g, g);
    }
    static constexpr float min_alpha = 0.1f;
    static glm::vec4 line_color(float weight) {
        float g = Network::sigmoid(weight);
        float a = (std::abs(0.5f-g) + min_alpha) * (1 / (0.5f + min_alpha));
        return glm::vec4(1-g, g, 0, a);
    }

    static void draw(Network & n, glm::vec2 position, glm::vec2 dimentions, GLFWE::Text::CharacterSet & text_writer) {
        std::vector<Network::Layer> & network = n.layers();

        unsigned int tallest_layer_size = 0;
        for (Network::Layer & l : network) if (tallest_layer_size < l.size()) tallest_layer_size = l.size();

        const int network_layer_count = network.size();
        const float horizontal_spacing = dimentions.x / (network_layer_count - 1);
        const float vertical_spacing = (tallest_layer_size == 1) ? 0 : dimentions.y / (tallest_layer_size - 1);

        const float node_radius = 20;
        const float line_thickness = 2;

        const auto horizontal_position = [position, horizontal_spacing](int index) -> float {
            return position.x + index * horizontal_spacing;
        };
        const auto vertical_position = [position, dimentions, vertical_spacing, tallest_layer_size](int layer_size, int index) -> float {
            return position.y + (dimentions.y) - (index + (tallest_layer_size - layer_size)/2.0f) * vertical_spacing;
        };

        // draw connecting lines
        int previous_layer_size = network[0].size();
        float prev_x = horizontal_position(0);

        for (int layer_i = 1; layer_i < network.size(); layer_i++) {
            Network::Layer & layer = network[layer_i];

            int layer_size = layer.size();
            float curr_x = horizontal_position(layer_i);

            // for each node in the previous layer
            for (int prev_i = 0; prev_i < previous_layer_size; prev_i++) {
                glm::vec2 prev_position = {prev_x, vertical_position(previous_layer_size, prev_i)};

                // connect to each node in the current layer
                for (int curr_i = 0; curr_i < layer_size; curr_i++) {
                    glm::vec2 curr_position = {curr_x, vertical_position(layer_size, curr_i)};

                    GLFWE::Shape::ShapeShader::set_color(line_color(layer[curr_i].weights[prev_i]));
                    GLFWE::Shape::Quad::draw_line(prev_position, curr_position, line_thickness);

                    // int weight = std::round(layer[curr_i].weights[prev_i] * 100);
                    // text_writer.render_string(std::to_string(weight), prev_position + (curr_position-prev_position)*glm::vec2(0.5, 0.5), 1, {0, 0, 0});
                }
            }

            previous_layer_size = layer_size;
            prev_x = curr_x;
        }

        // draw nodes on top
        for (unsigned int layer_i = 0; layer_i < network.size(); layer_i++) {
            int layer_size = network[layer_i].size();
            float curr_x = horizontal_position(layer_i);

            for (int curr_i = 0; curr_i < layer_size; curr_i++) {
                glm::vec2 position = {curr_x, vertical_position(layer_size, curr_i)};
                float node_value = network[layer_i].is_input_layer ? network[layer_i][curr_i].raw() : network[layer_i][curr_i].value();

                GLFWE::Shape::ShapeShader::set_color(node_color(node_value));
                GLFWE::Shape::Quad::draw_square(position - glm::vec2(node_radius, node_radius), glm::vec2(node_radius*2, node_radius*2));

                int node_value_zero_hundred = std::round(node_value * 100);
                text_writer.render_string(std::to_string(node_value_zero_hundred), position - glm::vec2{17, 7}, 1, {0, 0, 0});
                
                int node_bias = std::round(network[layer_i][curr_i].bias * 100);
                text_writer.render_string(std::to_string(node_bias), position - glm::vec2{17, 40}, 1, {0, 0, 0});
            }
        }
    }
};
}