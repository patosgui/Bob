SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PYTHONPATH="${SCRIPTPATH}/.." pytest ${SCRIPTPATH}