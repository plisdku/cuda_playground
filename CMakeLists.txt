############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
#           modified by Paul Hansen!!!!  yay
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(playground)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

#set (EXECUTABLE_OUTPUT_PATH "${CMAKE_BUILD_DIR}/bin/")
#message("exec output path is ${EXECUTABLE_OUTPUT_PATH}")
#file (MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/gcc; 
  -gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_35,code=compute_35;")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  # --------- Code-related things

  CUDA_ADD_EXECUTABLE(myPlayground main.cpp utils.hpp playground.cu)



else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()




