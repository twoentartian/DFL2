add_executable(generate_node_address generate_node_address.cpp)
target_link_libraries(generate_node_address OpenSSL::Crypto)

add_executable(print_rocksdb print_rocksdb.cpp)
target_link_libraries(print_rocksdb RocksDB::rocksdb glog::glog)

add_executable(block_db_to_json block_db_to_json.cpp)
target_link_libraries(block_db_to_json caffe Boost::serialization LZ4::lz4_static OpenSSL::Crypto RocksDB::rocksdb)

add_executable(dirichlet_distribution_config_generator dirichlet_distribution_config_generator.cpp)
target_link_libraries(dirichlet_distribution_config_generator Boost::serialization glog::glog)

add_executable(large_scale_simulation_generator large_scale_simulation_generator.cpp)
target_link_libraries(large_scale_simulation_generator Boost::serialization glog::glog)

#add_executable(remove_caffe_log remove_caffe_log.cpp)
#target_link_libraries(remove_caffe_log  Boost::serialization)
