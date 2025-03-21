#include <logger/logger.hpp>

#include <GLFWE/window.hpp>
#include <GLFWE/shader.hpp>
#include <GLFWE/shader_program.hpp>
#include <GLFWE/buffer.hpp>
#include <GLFWE/vertex_array.hpp>
#include <GLFWE/texture.hpp>

#include <GLFWE/text/character_set.hpp>

#include <chrono>
#include <ctime>

int main() {
    static constexpr Logger logger = Logger("MAIN");

    const glm::ivec2 window_dimentions = {800, 800};

    // create window
    auto & window = GLFWE::Window::create("Falling Everything", window_dimentions);

    while(GLFWE::Window::has_instance()) {
        window.update();
    }

    

    return 0;
}


