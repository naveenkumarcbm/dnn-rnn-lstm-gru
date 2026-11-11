# Assignment 2: Recurrent Models and Transformers

## Environment

Python 3.10+ and the pinned packages in `requirements.txt` are required. The notebooks use PyTorch 2.6.0, NumPy, pandas, Matplotlib, and scikit-learn. A CUDA GPU is strongly recommended; the specified full experiment set is intended for a Colab T4. The local environment now has the pinned dependencies and passed the quick unit checks, but the full experiment suite has not been run on a T4.

## Run

1. Install dependencies: `pip install -r requirements.txt`.
2. Open `problem1/problem1.ipynb` and `problem2/problem2.ipynb` in Jupyter or Colab. Both notebooks now embed their implementation and can be uploaded individually; they do not require the separate `solution.py` files.
3. Each notebook's default `RUN_FULL_EXPERIMENTS = False` path runs the implementation tests and parameter-count checks. Set it to `True` to download the datasets and run the complete experiments. Run the Problem 1 notebook first; Problem 2 independently uses the same HAR splitting and normalization method.
4. Run each notebook top to bottom. Training run rows are appended to root `results.csv`; plots are saved under each problem's `figures/`; the language-model run writes `problem2/generated_samples.txt`.

For an explanation of the model choices, equations, tensor shapes, and experiment code flow, see [learn.md](./learn.md).

The full run time should be measured on the submission T4 after enabling the complete sweeps. It is not reported as a measured value because the experiments have not yet been run in this environment. Full sweeps include 36 adding-problem runs, six HAR recurrent runs, 9 HAR Transformer runs, the character LM and its ablations; they may take several hours.

## Variant and limitations

The recurrent cells and scaled dot-product attention are implemented from scratch. Task 1.5(c) is interpreted as a gradient-clipping ablation on the LSTM adding task at T=200 because the supplied brief gives the rubric but no more specific ablation definition. The unmasked language model is intentionally included as an invalid leakage baseline. Current result and generated-sample files are explicit placeholders until the full notebooks are run; no performance results are claimed.

## AI use and references

AI assistance was used to draft and check the implementation structure. The student should review, run, interpret, and take responsibility for all code and reported results.

- UCI Machine Learning Repository, *Human Activity Recognition Using Smartphones*: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Andrej Karpathy, *tiny Shakespeare*: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
