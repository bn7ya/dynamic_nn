find_package(GTest QUIET)
if(NOT GTest_FOUND)
    return()
endif()

add_executable(enn_tests
    test_plateau_detector.cpp
    test_stability_monitor.cpp
    test_adaptive_lr_controller.cpp
    test_reversible_linear.cpp
    test_reversible_network.cpp
    test_adaptive_conv2d.cpp
    test_data_driven_config.cpp
    test_phase_controller.cpp
)
target_link_libraries(enn_tests PRIVATE enn_core GTest::gtest_main ${TORCH_LIBRARIES})
include(GoogleTest)
gtest_discover_tests(enn_tests)
