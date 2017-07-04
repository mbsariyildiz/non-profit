cmake_minimum_required(VERSION 2.8)
project(rgb-2-gray)

# to activate c++11
list(APPEND CMAKE_CXX_FLAGS "-std=c++0x")

# where headers are located
include_directories(include)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${CUDA_INCLUDE_DIRS})

# wilcard additions for all source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB CUDA_FILES "src/*.cu")

# load libs
find_package(CUDA REQUIRED)
set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)

# compile cuda files
set(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
#CUDA_COMPILE(CU_O ${CUDA_FILES})

# compile sources and link cuda objects to the executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_FILES})

# link opencv library
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
