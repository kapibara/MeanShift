cmake_minimum_required(VERSION 2.8)

project(MeanShift)

set(HOME ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_INSTALL_PREFIX ${HOME}/bin)

if(WIN32)
    set(DEV_BASE "C:/Data/Development/CPP/")
else()
    set(DEV_BASE "/home/kuznetso/Projects/CPP/")
    set(FLANN_ROOT "/home/kuznetso/bin/flann")
endif()


include_directories(${HOME}/src)
#using a template-only version of flann
include_directories(${FLANN_ROOT}/include)

link_directories(${FLANN_ROOT}/lib)

set(FLANN_LIBS flann flann_cpp hdf5)

set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_EXE_LINKER_FLAGS}")


#get_cmake_property(_variableNames VARIABLES)
#foreach(_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} /Od")

add_subdirectory(${HOME}/src)
