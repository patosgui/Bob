SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PYTHONPATH="${SCRIPTPATH}/.." pytest ${SCRIPTPATH} -k test_audio_client
