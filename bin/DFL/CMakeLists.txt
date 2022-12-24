add_executable(DFL DFL.cpp)
target_link_libraries(DFL ${DFL_LINKER_LIBS})

add_dependencies(DFL data_injector_mnist DFL_reputations)