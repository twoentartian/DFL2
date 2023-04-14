add_executable(TEST_ml_abs_temp_test temp_test.cpp)
#target_link_libraries(TEST_TensorFlow_Startup ${TENSORFLOW_LIB})
target_link_libraries(TEST_ml_abs_temp_test ${DFL_LINKER_LIBS})

add_executable(TEST_ml_abs_caffe test_caffe.cpp)
target_link_libraries(TEST_ml_abs_caffe ${DFL_LINKER_LIBS})

add_executable(TEST_ml_abs_memory_data_layer test_memory_data_layer.cpp)
target_link_libraries(TEST_ml_abs_memory_data_layer ${DFL_LINKER_LIBS})