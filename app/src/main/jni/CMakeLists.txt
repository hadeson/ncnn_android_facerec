project(Imageutil)
cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)

add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI}/libncnn.a)
get_filename_component(IMGUTIL_SRC_DIR  ${CMAKE_SOURCE_DIR}  ABSOLUTE)
get_filename_component (NCNN_JNI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ncnn  ABSOLUTE)
include_directories(${CMAKE_SOURCE_DIR}/ncnn/include
                    ${CMAKE_SOURCE_DIR})

if (ANDROID_ABI MATCHES "^armeabi-v7a$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=softfp -mfpu=neon -fopenmp -static-openmp")
elseif(ANDROID_ABI MATCHES "^arm64-v8a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ftree-vectorize -fopenmp -static-openmp")
endif()

# added
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

add_definitions(-DHEADER_FILE=${HEADER_FILE})
add_definitions(-DNAMESPACE=${NAMESPACE})
add_definitions(-DNCNN_INPUT=${NCNN_INPUT})
add_definitions(-DNCNN_OUTPUT=${NCNN_OUTPUT})

#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
                              -Wl,--allow-multiple-definition \
                              -Wl,--whole-archive -fPIE -v -fopenmp")

file(GLOB_RECURSE imgutil_sources ${IMGUTIL_SRC_DIR}/*.cc)
file(GLOB ncnn_jni_src ${NCNN_JNI_SOURCE_DIR}/*.cpp)

add_library(imgutil SHARED
            ${imgutil_sources})
target_include_directories(imgutil PRIVATE
                           ${CMAKE_SOURCE_DIR}
			   ${NCNN_JNI_SOURCE_DIR}/include
                           )

#add_library(imgutil SHARED ncnn_jni.cpp)
target_link_libraries(imgutil
                      android
                      log
                      jnigraphics
                      m
                      atomic
                      z)

#add_library(ncnn_jni SHARED ncnn/ncnn_jni.cpp)
file(GLOB FACE_SRC ncnn/*.cpp
                    ncnn/*.h)
set(FACE_COMPILE_CODE ${FACE_SRC})
add_library(ncnn_jni SHARED ${FACE_COMPILE_CODE})

target_link_libraries(ncnn_jni
                      ncnn
                      android
                      log
                      jnigraphics
                      m
                      atomic
                      z)

# Load OpenCV
set(OpenCV_DIR "/home/sonct/setup/OpenCV-android-sdk/sdk/native/jni/")
#set(OpenCV_DIR "C:\\Users\\caoth\\github_repos\\OpenCV-android-sdk\\sdk\\native\\jni")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
target_link_libraries(ncnn_jni ${OpenCV_LIBS})

