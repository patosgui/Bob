SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")/..
python3 -m black --line-length=80 --skip-magic-trailing-comma --target-version py310 --extend-exclude 'python-govee-api' ${SCRIPTPATH}
