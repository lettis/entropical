
cmake_minimum_required (VERSION 2.8)
project (transs)

# enable static linking by
#  cmake [...] -DCMATCH_STATIC_LINKING=ON
if(CMATCH_STATIC_LINKING)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif(CMATCH_STATIC_LINKING)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  # profiling
  #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p")
  # debugging
  #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
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

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(coords_file)

find_package(Boost 1.49 COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

#set (OpenCL_INCPATH "/usr/local/cuda/include")
#set (OpenCL_LIBPATH "/usr/local/cuda/lib64")

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

if (${DEBUG})
  message(STATUS "debug mode")
  #TODO: add debug flags to compiler
else()
  message(STATUS "production mode")
  add_definitions(-DNDEBUG)
endif()


if (${USE_OPENCL})
  message(STATUS "  Acceleration: using OpenCL")
  find_package(OpenCL REQUIRED)
  include_directories (${OPENCL_INCLUDE_DIRS})
  set (IM_SRC ${IM_SRC} densities_opencl.cpp
                        tools_opencl.cpp)
  set (IM_LIBS ${IM_LIBS} ${OPENCL_LIBRARIES})
  add_definitions(-DUSE_OPENCL)
elseif(${USE_CUDA})
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xcompiler -std=c++0x)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=compute_35)

  message(STATUS "  Acceleration: using CUDA")
  find_package(CUDA REQUIRED)
  include_directories (${CUDA_INCLUDE_DIRS})
  set (IM_SRC ${IM_SRC} densities_cuda.cu)
  set (IM_LIBS ${IM_LIBS} ${CUDA_LIBRARIES})
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

install(TARGETS im RUNTIME DESTINATION .)

