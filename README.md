# One time setup

python3 -m venv env

(PowerShell)
# Allow executing .ps1 files
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
# Allows the processes to have more permissions. Required for setting venv
Set-ExecutionPolicy Unrestricted -Scope Process

(Bash)
source env/bin/activate
(PowerShell)
.\env\Scripts\Activate.ps1 

(PowerShell) To verify if the right python is being picked up
gcm python

pip3 install -r requirements.txt

# To install the Govee API (after being in the venvv)
cd ./python-govee-api/

python3 ./setup.py install
