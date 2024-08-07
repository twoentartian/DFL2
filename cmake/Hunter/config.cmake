#hunter_config(Caffe VERSION ${HUNTER_Caffe_VERSION} CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})

hunter_config(Protobuf VERSION ${HUNTER_Protobuf_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(Boost VERSION ${HUNTER_Boost_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(gflags VERSION ${HUNTER_gflags_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(OpenCV VERSION ${HUNTER_OpenCV_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(OpenEXR VERSION ${HUNTER_OpenEXR_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(OpenBLAS VERSION ${HUNTER_OpenBLAS_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON -DNO_AVX512=1)

hunter_config(PNG VERSION ${HUNTER_PNG_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(jasper VERSION ${HUNTER_jasper_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(Jpeg VERSION ${HUNTER_Jpeg_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(ZLIB VERSION ${HUNTER_ZLIB_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)

hunter_config(rocksdb VERSION ${HUNTER_rocksdb_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(LZ4 VERSION ${HUNTER_lz4_VERSION} CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
