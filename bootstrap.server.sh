#!/usr/bin/env bash

ROOT=$(dirname "$(readlink -f "$0")")
VENV=${ROOT}/env/bin/activate

echo -e "\e[33mSourcing python venv from $VENV\e[0m"
source $VENV

echo -e "\e[33mStarting Bob...\e[0m"
./main.py