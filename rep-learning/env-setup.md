# Environment Setup

Needs Python. Tested with Python 3.14.

## Create the environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The folder is named `venv` so `.gitignore` skips it.

## GPU (optional)

The default `torch` wheel may be CPU only. For a CUDA build, pick the install command from
[pytorch.org](https://pytorch.org/get-started/locally/). Check it with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Without a GPU, training runs 40 epochs instead of 200.

## Jupyter kernel

In VS Code, select `venv/bin/python` as the notebook kernel. For Jupyter Lab or Notebook, register it first:

```bash
python -m ipykernel install --user --name rep-learning
```

## Data (optional)

`data/crm.csv` is included. To make it again:

```bash
python scripts/make_crm.py 3000 data/crm.csv
```

Then run the notebooks in order. They write to `outputs/`.

## Google Colab

Colab already has the packages. Each notebook runs on its own machine, so keep the repo on Google Drive so that `outputs/` from notebook 1 is there for notebook 2.

Note: For quick test use/run the merged notebook: [Merged Notebook](https://colab.research.google.com/github/rupakraj/DeepLearning/blob/main/rep-learning/building_embedding.ipynb)


1. Open a notebook:
   [01 Dataset preparation](https://colab.research.google.com/github/rupakraj/DeepLearning/blob/main/rep-learning/01_dataset_preparation.ipynb),
   [02 Model and training](https://colab.research.google.com/github/rupakraj/DeepLearning/blob/main/rep-learning/02_model_and_training.ipynb).
2. For GPU: **Runtime → Change runtime type → T4 GPU**.
3. Add this cell at the top and run it first:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive
![ -d DeepLearning ] || git clone https://github.com/rupakraj/DeepLearning.git
%cd DeepLearning/rep-learning
```

