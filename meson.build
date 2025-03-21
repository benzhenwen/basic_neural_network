project('network', 'cpp', 'c',
        default_options: 'default_library=static')

src_files = ['main.cpp']

logger_dep = dependency('logger', fallback: ['logger', 'logger_dep'])
glfwe_dep = dependency('glfwe', fallback: ['glfwe', 'glfwe_dep'])

subdir('fonts')
subdir('shaders')

executable('network', src_files,
    win_subsystem: 'windows',
    dependencies: [logger_dep, glfwe_dep])