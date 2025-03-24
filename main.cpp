#include <logger/logger.hpp>

#include <GLFWE/window.hpp>

#include <GLFWE/text/character_set.hpp>
#include <GLFWE/shape/quad.hpp>

#include <src/network.hpp>
#include <src/network_drawer.hpp>

#include <chrono>
#include <ctime>

int main() {
    static constexpr Logger logger = Logger("MAIN");

    const glm::ivec2 window_dimentions = {1200, 800};

    // create window
    auto & window = GLFWE::Window::create("network", window_dimentions);
    auto char_set = GLFWE::Text::CharacterSet("fonts/arial.ttf", 20);

    auto network = BN::Network(2);
    network.append_layers(3, 1);
    network.append_layers(2);
    network.randomize();

    window.clear_color();
    BN::NetworkDrawer::draw(network, {100, 100}, {1000, 600}, char_set);
    window.swap_buffers();

    std::chrono::time_point old = std::chrono::steady_clock::now();

    int i = 0;
    while(GLFWE::Window::has_instance()) {

        if (std::chrono::steady_clock::now() - old > std::chrono::duration(std::chrono::milliseconds(50))) {
            old = std::chrono::steady_clock::now();

            for (int j = 0; j < 1; j++) {
                bool a = i   % 2 == 0;
                bool b = i/2 % 2 == 0;
                i++;

                bool c = a || b;
                bool d = a && b;

                network.input_layer()[0].raw_value = a ? 1 : 0;
                network.input_layer()[1].raw_value = b ? 1 : 0;
                network.calculate();

                logger << a << " " << b << " -> " << network.output_layer()[0].value() << " " << network.output_layer()[1].value();

                std::vector<float> expected_output = {c ? 1.0f : 0.0f, d ? 1.0f : 0.0f};
                network.backpropagate(expected_output);
            }

            network.calculate();


            window.clear_color();
            BN::NetworkDrawer::draw(network, {100, 100}, {1000, 600}, char_set);
            window.swap_buffers();
        }


        window.update();
    }

    return 0;
}


