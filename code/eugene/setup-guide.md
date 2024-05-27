1. Open bash shell in this directory (````tiny-transformers/code/eugene```)
2. Create Conda environment (```conda create --name pretrain```)
3. Create symbolic link from ```common``` directory into this one (```ln -s ../common common```)
4. Activate the environment (```conda activate pretrain```)
5. Install PyTorch with correct CUDA version (e.g. ```pip3 install torch --index-url https://download.pytorch.org/whl/cu121 ```)
6. Install other dependencies (```pip3 install -r requirements.txt```)

