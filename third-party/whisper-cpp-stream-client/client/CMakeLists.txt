
# SDL2
find_package(SDL2 REQUIRED)

# string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)
# 
# message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
# message(STATUS "SDL2_LIBRARIES    = ${SDL2_LIBRARIES}")
# 
# # common-sdl
# set(TARGET common-sdl)
# 
# add_library(${TARGET} STATIC
#     ${WHISPER_EXAMPLES_DIR}/common-sdl.h
#     ${WHISPER_EXAMPLES_DIR}/common-sdl.cpp
#     )
# 
# include(DefaultTargetOptions)
# 
# target_include_directories(${TARGET} PUBLIC  ${SDL2_INCLUDE_DIRS})
# target_link_libraries     (${TARGET} PRIVATE ${SDL2_LIBRARIES})
# 
# set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

## the client

set(TARGET whisper-cpp-stream-client)
add_executable(${TARGET} main.cpp)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE "${CMAKE_SOURCE_DIR}/whisper.cpp/examples")

target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)