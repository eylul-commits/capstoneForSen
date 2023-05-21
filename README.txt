1. You need to download requirements to your environment (local or conda environment).
2. You need to install dlib using this command -> "pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl"
3. You need to install torchvision using this command -> "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
4. We have a FaceModel class in getFaceData.py, you can utilize it's function "getFaceData" by  providing an image, you can check "modelExample.py" for an example usage.