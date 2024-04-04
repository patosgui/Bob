set +x

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
ROOTPATH=$(dirname "$SCRIPTPATH")
PYTHONPATH="${ROOTPATH}/.." python3 ${ROOTPATH}/main.py wav --wav-file ${SCRIPTPATH}/test.wav