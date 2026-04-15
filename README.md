# CtEcho Runner

This folder is a minimal, GitHub-ready subset for one task:

1. Load a split `.npz`.
2. Extract CtEcho features.
3. Train an RBF SVM.
4. Report accuracy, classification report, and confusion matrix.

Expected `.npz` fields:

- Train data: `x_train_irregular` or `X_train_irregular` or `x_train` or `X_train`
- Test data: `x_test_irregular` or `X_test_irregular` or `x_test` or `X_test`
- Train timestamps: `timestamps_train` or `t_train` or `timestep_train` or `timesteps_train`
- Test timestamps: `timestamps_test` or `t_test` or `timestep_test` or `timesteps_test`
- Train labels: `y_train` or `labels_train` or `train_labels`
- Test labels: `y_test` or `labels_test` or `test_labels` or `true_labels`

Run:

```bash
python run.py path/to/split_data.npz
```

Environment setup:

1. Create a virtual environment (Python 3.11):

```bash
conda create --prefix="./pure_py311" python=3.11
```

2. Activate the virtual environment:

```bash
conda activate ./pure_py311/
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```


Main dependencies:

- python == 3.11
- numpy == 2.1.2
- torch == 2.6.0
- torchcde == 0.2.5
- torchdiffeq == 0.2.5
- scikit-learn == 1.6.1
- scipy == 1.17.1
- h5py == 3.16.0
- matplotlib == 3.10.8
- pandas == 3.0.2
- tqdm == 4.67.3
