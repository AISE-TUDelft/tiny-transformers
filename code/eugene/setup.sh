ENV_NAME="venv"

if  [ -d "$ENV_NAME" ] ; then
    echo "Directory $ENV_NAME already exists, not installing new virtual environment";
else
    python -m venv "$ENV_NAME"
fi

source "$ENV_NAME/Scripts/activate";

# Update pip & install dependencies
ln -s ../common common
python -m pip install --upgrade pip 
# Install all depedencies for embedded, except for typing
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade --ignore-installed  -r requirements.txt
