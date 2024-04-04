SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PYTHONPATH="${SCRIPTPATH}/.." python3 -m pytest ${SCRIPTPATH}
