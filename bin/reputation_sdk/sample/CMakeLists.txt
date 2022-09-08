add_library (reputation_api_sample SHARED "sample_reputation.cpp")
target_link_libraries (reputation_api_sample ${DFL_LINKER_LIBS})

add_library (reputation_FedAvg SHARED "FedAvg_reputation.cpp")
target_link_libraries (reputation_FedAvg ${DFL_LINKER_LIBS})

add_library (reputation_HalfFedAvg SHARED "HalfFedAvg_reputation.cpp")
target_link_libraries (reputation_HalfFedAvg ${DFL_LINKER_LIBS})

add_library (reputation_ML_hcluster SHARED "hcluster_reputation.cpp")
target_link_libraries (reputation_ML_hcluster ${DFL_LINKER_LIBS})

add_library (reputation_001 SHARED "reputation_001.cpp")
target_link_libraries (reputation_001 ${DFL_LINKER_LIBS})

add_library (reputation_005 SHARED "reputation_005.cpp")
target_link_libraries (reputation_005 ${DFL_LINKER_LIBS})
