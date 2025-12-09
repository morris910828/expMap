cmake_minimum_required(VERSION 3.10)
project(ExpMap)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Glad
add_subdirectory(external/glad)

# GLFW
add_subdirectory(external/glfw)

# ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
file(GLOB IMGUI_SOURCES
    ${IMGUI_DIR}/*.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_library(imgui ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

set(ASSIMP_BUILD_TESTS OFF)

# Assimp
add_subdirectory(external/assimp)

include_directories(
    external/glad/include
    external/glfw/include
    external/imgui
    external/imgui/backends
    external/assimp/include
    external/glm
    src
)

# ⭐ Fix: recursively include all .cpp / .h in src/
file(GLOB_RECURSE SRC_FILES
    src/*.cpp
    src/*.h
)

add_executable(ExpMap
    ${SRC_FILES}
)

target_link_libraries(ExpMap
    glad
    glfw
    imgui
    assimp
    opengl32
)
