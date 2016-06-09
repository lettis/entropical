
cmake_minimum_required (VERSION 2.8)
project (transs)

# enable static linking by
#  cmake [...] -DCMATCH_STATIC_LINKING=ON
if(CMATCH_STATIC_LINKING)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif(CMATCH_STATIC_LINKING)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
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
find_package(OpenCL REQUIRED)
include_directories (${OPENCL_INCLUDE_DIRS})

add_executable(im im.cpp
                  im_help.cpp

                  transs.cpp
                  transs_opencl.cpp
                  mi.cpp
                  dens.cpp
                  negs.cpp
                  amise.cpp
                  hestimate.cpp

                  densities.cpp
                  tools.cpp
                  tools_opencl.cpp)

target_link_libraries(im
                      ${Boost_LIBRARIES}
                      ${OPENCL_LIBRARIES}
                      coords_file)

install(TARGETS im RUNTIME DESTINATION .)

