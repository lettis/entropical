
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
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  # warnings
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-declarations")
else()
  message(SEND_ERROR "sorry, only GCC is supported.
                      if you want to use another compiler, adapt CMakeLists.txt and try at your own risk.")
endif()


add_subdirectory(coords_file)

find_package(Boost 1.49 COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

add_executable(transs transs.cpp tools.cpp)
target_link_libraries(transs ${Boost_LIBRARIES} coords_file)

install(TARGETS transs RUNTIME DESTINATION .)

