add_executable(generate_node_address generate_node_address.cpp)
target_link_libraries(generate_node_address ${DFL_LINKER_LIBS})

add_executable(print_rocksdb print_rocksdb.cpp)
target_link_libraries(print_rocksdb ${DFL_LINKER_LIBS})

add_executable(block_db_to_json block_db_to_json.cpp)
target_link_libraries(block_db_to_json ${DFL_LINKER_LIBS})

add_executable(dirichlet_distribution_config_generator dirichlet_distribution_config_generator.cpp)
target_link_libraries(dirichlet_distribution_config_generator ${DFL_LINKER_LIBS})

add_executable(large_scale_simulation_generator large_scale_simulation_generator.cpp)
target_link_libraries(large_scale_simulation_generator ${DFL_LINKER_LIBS})

add_executable(large_scale_simulation_generator_social_network large_scale_simulation_generator_social_network.cpp)
target_link_libraries(large_scale_simulation_generator_social_network ${DFL_LINKER_LIBS})

add_executable(large_scale_simulation_generator_maksim large_scale_simulation_generator_maksim.cpp)
target_link_libraries(large_scale_simulation_generator_maksim ${DFL_LINKER_LIBS})

add_executable(inject_interval_generator inject_interval_generator.cpp)
target_link_libraries(inject_interval_generator ${DFL_LINKER_LIBS})

add_executable(maksim_topology_generator maksim_topology_generator.cpp)
target_link_libraries(maksim_topology_generator ${DFL_LINKER_LIBS})

add_executable(topology_remove_receive topology_remove_receive.cpp)
target_link_libraries(topology_remove_receive ${DFL_LINKER_LIBS})

add_executable(model_util model_util.cpp)
target_link_libraries(model_util ${DFL_LINKER_LIBS})

add_executable(model_remove_weights_analysis model_remove_weights_analysis.cpp)
target_link_libraries(model_remove_weights_analysis ${DFL_LINKER_LIBS})

add_executable(calculate_model_fusion_accuracy calculate_model_fusion_accuracy.cpp)
target_link_libraries(calculate_model_fusion_accuracy ${DFL_LINKER_LIBS})
add_custom_command(TARGET calculate_model_fusion_accuracy POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/calculate_model_fusion_accuracy_set_env.sh ${CMAKE_CURRENT_BINARY_DIR}/calculate_model_fusion_accuracy_set_env.sh COMMENT "copy env script to build dir")

# find CUDA
if (NOT CPU_ONLY)
include(CheckLanguage)
check_language(CUDA)
set(ANALYZE_MODEL_SRC)
set(ANALYZE_MODEL_LINK)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "build analyze_models with CUDA")
    add_compile_definitions(ANALYZE_MODEL_USE_CUDA=1)
    set(CMAKE_CUDA_ARCHITECTURES 52)
    enable_language(CUDA)
    list(APPEND ANALYZE_MODEL_SRC analyze_models.cu)
    list(APPEND ANALYZE_MODEL_LINK ${CUDA_LIBRARIES})
else ()
    message(STATUS "build analyze_models without CUDA")
    add_compile_definitions(ANALYZE_MODEL_USE_CUDA=0)
endif ()
list(APPEND ANALYZE_MODEL_SRC analyze_models.cpp)
add_executable(analyze_models ${ANALYZE_MODEL_SRC})
target_link_libraries(analyze_models ${DFL_LINKER_LIBS} ${ANALYZE_MODEL_LINK})
if (CMAKE_CUDA_COMPILER)
    target_compile_options(analyze_models PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --generate-line-info --use_fast_math -O3>)
endif ()
endif ()
