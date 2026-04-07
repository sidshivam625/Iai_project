import os
import zipfile
import json

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            if '__pycache__' in root or '.venv' in root or 'artifacts' in root:
                continue
            file_path = os.path.join(root, file)
            ziph.write(file_path, os.path.relpath(file_path, '.'))

print("Creating kaggle_upload.zip...")
with zipfile.ZipFile('kaggle_upload.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for folder in ['src', 'data', 'Benign data']:
        if os.path.exists(folder):
            zipdir(folder, zipf)
    for file in ['train.py', 'requirements.txt']:
        if os.path.exists(file):
            zipf.write(file)
print("Done creating zip.")

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Train Prompt Injection Detector on Kaggle\n",
    "1. In Kaggle, click **Add Input** (or Add Data) -> **Upload**.\n",
    "2. Upload `kaggle_upload.zip` and give it a name (e.g., `prompt-injection-data`).\n",
    "3. Set your Notebook backend to **GPU T4x2** or **P100**.\n",
    "4. Change `YOUR_DATASET_NAME_HERE` in the cell below to the name you chose.\n",
    "5. Run the cells!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import shutil\n",
    "\n",
    "# Change this to whatever you named your dataset on upload!\n",
    "DATASET_NAME = 'prompt-injection-data' \n",
    "\n",
    "!cp /kaggle/input/{DATASET_NAME}/kaggle_upload.zip /kaggle/working/\n",
    "!unzip -q -o /kaggle/working/kaggle_upload.zip -d /kaggle/working/project\n",
    "os.chdir('/kaggle/working/project')\n",
    "\n",
    "# Install requirements (Kaggle already has torch and transformers, but just in case)\n",
    "!pip install -r requirements.txt -q\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run training! (Using batch size 64 since Kaggle GPUs have up to 16GB VRAM)\n",
    "!python train.py --epochs 2 --batch-size 64 --output-dir /kaggle/working/artifacts --use-cuda\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Zip the artifacts so you can download the trained models\n",
    "os.chdir('/kaggle/working/')\n",
    "!zip -r -q trained_models_artifacts.zip artifacts\n",
    "print('Training complete! Download trained_models_artifacts.zip from the side panel under Output -> /kaggle/working')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("kaggle_training_notebook.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Created kaggle_training_notebook.ipynb!")
