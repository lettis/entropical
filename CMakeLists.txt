
cmake_minimum_required (VERSION 2.8)
project (transs)

# enable static linking
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  # optimization
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
  # warnings
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-declarations")
else()
  message(SEND_ERROR "sorry, only GCC is supported.
                      if you want to use another compiler,
                      adapt CMakeLists.txt and try at your own risk.")
endif()

if (${DEBUG})
  message(STATUS "debug mode")
  # profiling
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p")
  # debugging
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
  message(STATUS "production mode")
  add_definitions(-DNDEBUG)
endif()


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(coords_file)

find_package(Boost 1.49 COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

set(IM_SRC im.cpp
           im_help.cpp
           mi.cpp
           transs.cpp
           dens.cpp
           negs.cpp
           amise.cpp
           hestimate.cpp
           tools.cpp)

set (IM_LIBS ${Boost_LIBRARIES} coords_file)


if(${USE_CUDA})
  message(STATUS "  Acceleration: using CUDA")
  find_package(CUDA REQUIRED)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS                     "-std=c++11")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ";-Xcompiler -fopenmp")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ";-O3;-DVERBOSE")
  set (IM_SRC ${IM_SRC} densities_cuda.cu)
  set (IM_LIBS ${IM_LIBS} ${CUDA_LIBRARIES})
  include_directories (${CUDA_INCLUDE_DIRS})
  add_definitions(-DUSE_CUDA)
else()
  message(STATUS "  Acceleration: using OpenMP")
  set (IM_SRC ${IM_SRC} densities_omp.cpp)
endif()

if(${USE_CUDA})
  cuda_add_executable(im ${IM_SRC})
else()
  add_executable(im ${IM_SRC})
endif()

target_link_libraries(im ${IM_LIBS})

install(TARGETS im DESTINATION .)

