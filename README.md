# Usage

## Setup the environment
```
python3 -m venv .venv
pip install -r requiremments.txt
```

Do `pre-commit install` to install the pre-commit hooks

## Add a configuration file

The configuration file specifies the models to use as well as any accessories
that should be used by the application.

Example:
```
conversation_model:
  mistral:
    api_key: "<your_api_key>"

accessories:
  hue_bridge:
    ip: <ip_address_of_your_bridge>
```

## Execute the application
```
python3 main.py microphone
```
