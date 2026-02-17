cmake_minimum_required(VERSION 3.10)

# set the project name
project(GTasb3D VERSION 1.0 LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
#set(CMAKE_CXX_FLAGS "-std=c++-11 -O3")
set(CAMEK_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -I ~/include/armadillo/include/ -L ~/lib/lapack/ -L ~/include/armadillo/lib64/")

# add the executable
file (GLOB_RECURSE GTA_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_executable(GTasb3D ${GTA_SOURCES})

# find package
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(MPI REQUIRED)
if(OpenMP_CXX_FOUND)
	target_link_libraries(GTasb3D PRIVATE OpenMP::OpenMP_CXX ${ARMADILLO_LIBRARIES})
endif()
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
# Dont inlcude any mpicxx (causes linker to fail)
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)


# add directory to the list of paths to search for include files
target_include_directories(GTasb3D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find and link the p4est library
# Prefer absolute paths via find_library over -l/name guessing
set(P4EST_HINT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../p4est/local")
find_library(P4EST_LIB NAMES p4est PATHS "${P4EST_HINT_DIR}/lib" NO_DEFAULT_PATH)
find_library(SC_LIB NAMES sc PATHS "${P4EST_HINT_DIR}/lib" NO_DEFAULT_PATH)

if(NOT P4EST_LIB OR NOT SC_LIB)
    message(FATAL_ERROR "Could not find p4est/sc static libraries under ${P4EST_HINT_DIR}/lib. Build/install p4est to ${P4EST_HINT_DIR} or adjust P4EST_HINT_DIR.")
endif()

target_include_directories(GTasb3D PRIVATE "${P4EST_HINT_DIR}/include")
target_link_libraries(GTasb3D PRIVATE ${MPI_LIBRARIES} ${P4EST_LIB} ${SC_LIB})
