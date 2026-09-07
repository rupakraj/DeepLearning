# DeepLearning

Collection of scripts I use in Classroom


## Setup

```bash
python -m venv .venv
source ./.venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name deeplearning
```

## Run in VS Code

1. Open this folder in VS Code, install the **Jupyter** and **Python** extensions.
2. Open `*.ipynb`.
3. Select the `deeplearning` kernel (or `.venv`) via the kernel picker (top-right).
4. Run cells with `Shift+Enter`, or **Run All**.


## Collections

| SN | Topic | Script |
| --- | ---| ---|
|1 | Introduction to Pytorch (with toy example) | [GO](pytorch_toy_example.ipynb) |
|2 | Dataset splits, Overfit/Underfit and Dropout/L1/L2 regularization | [GO](overfit_underfit_regularization.ipynb) |
|3 | Limitation of Gradient Descent | [GO](limitation_of_gradient.ipynb) |
|4 | Adaptive Learning Rates: AdaGrad, RMSProp, Adam | [GO](adaptive_learning_rates.ipynb) |
|5 | Why do we need CNNs? DNN limitations vs. Convolutions (FashionMNIST) | [GO](need_for_cnn.ipynb) |

---
__Note__: This list will gradually grow.  Hopefully in 2-3 months :-)
