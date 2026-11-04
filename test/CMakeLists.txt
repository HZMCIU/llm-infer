enable_testing()

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main)
    target_link_libraries(${test_name} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    target_link_libraries(${test_name} PRIVATE src_objects)
endforeach()

include(GoogleTest)
gtest_discover_tests(kernel_test)
