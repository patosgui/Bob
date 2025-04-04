
cmake_minimum_required(VERSION 3.29)
project("whisper-cpp-stream-client" C CXX)
project("whisper-cpp-stream-client" VERSION 0.0.1)

# Required to get data from the system microphone
# TODO: Check that sdl2-dev is installed
option(WHISPER_SDL2 ON)
option(WHISPER_BUILD_EXAMPLES ON)

set(WHISPER_BUILD_EXAMPLES ON CACHE BOOL "whisper: build examples" FORCE)
set(WHISPER_SDL2 ON CACHE BOOL "whisper: build examples" FORCE)
add_subdirectory(whisper.cpp)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/cmake")

#set(WHISPER_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/examples")

add_subdirectory(client)