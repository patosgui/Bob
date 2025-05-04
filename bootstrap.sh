#!/usr/bin/env bash

ROOT=$(dirname "$(readlink -f "$0")")
VENV=${ROOT}/.venv/bin/activate

echo -e "\e[33mSourcing python venv from $VENV\e[0m"
source $VENV

echo -e "\e[33mStarting Bob...\e[0m"
./third-party/whisper-cpp-stream-client/build/bin/whisper-cpp-stream-client -m third-party/whisper-cpp-stream-client/whisper.cpp/models/ggml-base.en.bin -t 8 -vth 0.6 --step 0 --length 5000 -c 0 &
./main.py microphone
