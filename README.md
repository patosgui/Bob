# One time setup

python3 -m venv env

source env/bin/activate

pip3 install -r requirements.txt

# To install the Govee API (after being in the venvv)
cd ./python-govee-api/

python3 ./setup.py install
