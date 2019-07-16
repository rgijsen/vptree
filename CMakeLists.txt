cmake_minimum_required(VERSION 3.10)
project(VPTree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -w")

set(CMAKE_VERBOSE_MAKEFILE ON)

#put all in one directory...
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#====================================
macro(copy_file_to_target_dir arg0 arg1)
    add_custom_command(TARGET ${ARGV0} PRE_BUILD        # Adds a post-build event to project name
        COMMAND ${CMAKE_COMMAND} -E copy_if_different   # which executes "cmake - E copy_if_different..."
            ${ARGV1}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>                  # <--this is out-file path
    )
endmacro(copy_file_to_target_dir)
#=========================

add_subdirectory(src)
