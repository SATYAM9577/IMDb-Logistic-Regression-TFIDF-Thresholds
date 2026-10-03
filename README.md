# IMDb Sentiment Classification with Logistic Regression

This assignment compares Logistic Regression using unigram TF-IDF and
unigram-plus-bigram TF-IDF on the official IMDb Large Movie Review Dataset.
It also measures threshold effects and examines five borderline test reviews.

## Run

```powershell
py -m pip install -r requirements.txt
py assignment.py
```

The script downloads the official train and test parquet splits on first run.
The test split is the same 25,000-review split used in the related Q1 and Q2
assignments. Dataset files are stored under `data/` and excluded from Git.

## Outputs

- `results/model_comparison.csv`: accuracy, precision, recall, and F1 for both
  representations
- `results/confusion_matrix.png` and `.csv`: confusion matrix for the higher-F1
  model
- `results/threshold_metrics.csv` and `threshold_precision_recall.png`: metrics
  and plot for thresholds 0.3, 0.5, and 0.7
- `results/borderline_examples.csv`: five test rows closest to probability 0.5
- `results/assignment_report.md`: results, examples, and discussion

Full review text is not reproduced in committed files; the report uses
paraphrased descriptions and model evidence to discuss borderline language
patterns.

## Experiment details

- Fixed official IMDb train/test split; labels are 0 (negative) and 1 (positive).
- TF-IDF uses lowercase text, Unicode accent stripping, `min_df=2`,
  `max_df=0.95`, and sublinear term frequency.
- Logistic Regression uses `C=1`, `max_iter=2000`, `liblinear`, and a fixed
  random state. Both feature extractors are fit on training data only.
- Precision, recall, and F1 use the positive class as the positive label.
- Threshold analysis uses probabilities from the higher-F1 of the two models.

Dataset citation: Maas, A. L. et al. (2011), *Learning Word Vectors for
Sentiment Analysis*, ACL. Dataset source:
[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).
