SCRIPT=$(readlink -f "$0")
REPO_ROOT=$(dirname "$SCRIPT")/..
python3 -m black --line-length=80 --skip-magic-trailing-comma --target-version py310 --extend-exclude 'python-govee-api' ${REPO_ROOT}
fixit lint ${REPO_ROOT}
