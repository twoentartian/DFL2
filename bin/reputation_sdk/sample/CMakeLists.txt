if(UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_DEBUG_POSTFIX "")

add_library (reputation_api_sample SHARED "sample_reputation.cpp")
target_link_libraries (reputation_api_sample ${DFL_LINKER_LIBS})

add_library (reputation_FedAvg SHARED "FedAvg_reputation.cpp")
target_link_libraries (reputation_FedAvg ${DFL_LINKER_LIBS})

add_library (reputation_HalfFedAvg SHARED "HalfFedAvg_reputation.cpp")
target_link_libraries (reputation_HalfFedAvg ${DFL_LINKER_LIBS})

add_library (reputation_ML_hcluster SHARED "hcluster_reputation.cpp")
target_link_libraries (reputation_ML_hcluster ${DFL_LINKER_LIBS})

add_library (50_train_50_average_based_on_accuracy SHARED "50_train_50_average_based_on_accuracy.cpp")
target_link_libraries (50_train_50_average_based_on_accuracy ${DFL_LINKER_LIBS})

add_library (reputation_001 SHARED "reputation_001.cpp")
target_link_libraries (reputation_001 ${DFL_LINKER_LIBS})

add_library (reputation_005 SHARED "reputation_005.cpp")
target_link_libraries (reputation_005 ${DFL_LINKER_LIBS})

add_library (reputation_100_training_0_averaging SHARED "100_training_0_averaging.cpp")
target_link_libraries (reputation_100_training_0_averaging ${DFL_LINKER_LIBS})

add_library (reputation_98_training_2_averaging SHARED "98_training_2_averaging.cpp")
target_link_libraries (reputation_98_training_2_averaging ${DFL_LINKER_LIBS})

add_library (reputation_96_training_4_averaging SHARED "96_training_4_averaging.cpp")
target_link_libraries (reputation_96_training_4_averaging ${DFL_LINKER_LIBS})

add_library (reputation_94_training_6_averaging SHARED "94_training_6_averaging.cpp")
target_link_libraries (reputation_94_training_6_averaging ${DFL_LINKER_LIBS})

add_library (reputation_92_training_8_averaging SHARED "92_training_8_averaging.cpp")
target_link_libraries (reputation_92_training_8_averaging ${DFL_LINKER_LIBS})

add_library (reputation_90_training_10_averaging SHARED "90_training_10_averaging.cpp")
target_link_libraries (reputation_90_training_10_averaging ${DFL_LINKER_LIBS})

add_library (reputation_80_training_20_averaging SHARED "80_training_20_averaging.cpp")
target_link_libraries (reputation_80_training_20_averaging ${DFL_LINKER_LIBS})

add_library (reputation_75_training_25_averaging SHARED "75_training_25_averaging.cpp")
target_link_libraries (reputation_75_training_25_averaging ${DFL_LINKER_LIBS})

add_library (reputation_70_training_30_averaging SHARED "70_training_30_averaging.cpp")
target_link_libraries (reputation_70_training_30_averaging ${DFL_LINKER_LIBS})

add_library (reputation_60_training_40_averaging SHARED "60_training_40_averaging.cpp")
target_link_libraries (reputation_60_training_40_averaging ${DFL_LINKER_LIBS})

add_library (reputation_50_training_50_averaging SHARED "50_training_50_averaging.cpp")
target_link_libraries (reputation_50_training_50_averaging ${DFL_LINKER_LIBS})

add_library (reputation_40_training_60_averaging SHARED "40_training_60_averaging.cpp")
target_link_libraries (reputation_40_training_60_averaging ${DFL_LINKER_LIBS})

add_library (reputation_30_training_70_averaging SHARED "30_training_70_averaging.cpp")
target_link_libraries (reputation_30_training_70_averaging ${DFL_LINKER_LIBS})

add_library (reputation_25_training_75_averaging SHARED "25_training_75_averaging.cpp")
target_link_libraries (reputation_25_training_75_averaging ${DFL_LINKER_LIBS})

add_library (reputation_20_training_80_averaging SHARED "20_training_80_averaging.cpp")
target_link_libraries (reputation_20_training_80_averaging ${DFL_LINKER_LIBS})

add_library (reputation_10_training_90_averaging SHARED "10_training_90_averaging.cpp")
target_link_libraries (reputation_10_training_90_averaging ${DFL_LINKER_LIBS})

add_library (reputation_8_training_92_averaging SHARED "8_training_92_averaging.cpp")
target_link_libraries (reputation_8_training_92_averaging ${DFL_LINKER_LIBS})

add_library (reputation_6_training_94_averaging SHARED "6_training_94_averaging.cpp")
target_link_libraries (reputation_6_training_94_averaging ${DFL_LINKER_LIBS})

add_library (reputation_4_training_96_averaging SHARED "4_training_96_averaging.cpp")
target_link_libraries (reputation_4_training_96_averaging ${DFL_LINKER_LIBS})

add_library (reputation_2_training_98_averaging SHARED "2_training_98_averaging.cpp")
target_link_libraries (reputation_2_training_98_averaging ${DFL_LINKER_LIBS})

add_custom_target(DFL_reputations DEPENDS reputation_api_sample reputation_FedAvg reputation_HalfFedAvg reputation_ML_hcluster
        reputation_001 reputation_005 50_train_50_average_based_on_accuracy
        reputation_100_training_0_averaging
        reputation_98_training_2_averaging reputation_96_training_4_averaging
        reputation_94_training_6_averaging reputation_92_training_8_averaging
        reputation_90_training_10_averaging reputation_80_training_20_averaging
        reputation_75_training_25_averaging
        reputation_70_training_30_averaging reputation_60_training_40_averaging
        reputation_50_training_50_averaging
        reputation_40_training_60_averaging reputation_30_training_70_averaging
        reputation_25_training_75_averaging
        reputation_20_training_80_averaging reputation_10_training_90_averaging
        reputation_8_training_92_averaging reputation_6_training_94_averaging
        reputation_4_training_96_averaging reputation_2_training_98_averaging
        )