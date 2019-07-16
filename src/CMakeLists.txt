cmake_minimum_required(VERSION 3.10)
project(VPTree)

set(SOURCES
		main.cpp
		vptree_tests.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

copy_file_to_target_dir(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/resources/cubic10_pbc_benchmark.txt)