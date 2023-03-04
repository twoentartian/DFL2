add_executable(DFL_simulator_mt simulator_mt.cpp)
target_link_libraries(DFL_simulator_mt ${DFL_LINKER_LIBS})

add_dependencies(DFL_simulator_mt DFL_reputations large_scale_simulation_generator large_scale_simulation_generator_social_network dirichlet_distribution_config_generator inject_interval_generator)
add_custom_command(TARGET DFL_simulator_mt POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/run_simulator.sh ${CMAKE_CURRENT_BINARY_DIR}/run_simulator.sh COMMENT "copy simulation script to build dir")