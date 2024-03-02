# Create a virtual environment
python3 -m venv env
# Allow the venv activate script to overwrite the environment variables
Set-ExecutionPolicy Unrestricted -Scope Process
# Activate the venv
& $PSScriptRoot\env\Scripts\Activate.ps1

Write-Output "Location of the python binary"
gcm python

# Install the requirements
pip3 install -r requirements.txt