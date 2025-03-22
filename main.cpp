#include <logger/logger.hpp>

#include <GLFWE/window.hpp>
#include <GLFWE/shader.hpp>
#include <GLFWE/shader_program.hpp>
#include <GLFWE/buffer.hpp>
#include <GLFWE/vertex_array.hpp>
#include <GLFWE/texture.hpp>

#include <GLFWE/text/character_set.hpp>
#include <GLFWE/shape/quad.hpp>

#include <chrono>
#include <ctime>

int main() {
    static constexpr Logger logger = Logger("MAIN");

    const glm::ivec2 window_dimentions = {800, 800};

    // create window
    auto & window = GLFWE::Window::create("network", window_dimentions);

    auto char_set = GLFWE::Text::CharacterSet("fonts/arial.ttf", 100);

    while(GLFWE::Window::has_instance()) {

        window.clear_color();

        GLFWE::Shape::ShapeShader::set_color({0, 1, 0});
        GLFWE::Shape::Quad::draw_square({250, 250}, {300, 300});
        GLFWE::Shape::ShapeShader::set_color({1, 0, 0});
        GLFWE::Shape::Quad::draw_line({100, 100}, {700, 300}, 15);

        char_set.render_string("meow", {10, 400}, 1.0f, {0.3f, 0.0f, 1.0f});
        
        window.swap_buffers();

        window.update();

    }

    

    return 0;
}


