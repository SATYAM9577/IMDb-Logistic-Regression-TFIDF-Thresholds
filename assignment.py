from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SPLITS = {
    "train": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet",
}
THRESHOLDS = (0.3, 0.5, 0.7)


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, url in SPLITS.items():
        destination = DATA_DIR / f"imdb_{split}.parquet"
        if not destination.is_file():
            print(f"Downloading IMDb {split} split...")
            urllib.request.urlretrieve(url, destination)


def read_split(split: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_parquet(DATA_DIR / f"imdb_{split}.parquet", columns=["text", "label"])
    return frame["text"].astype(str).tolist(), frame["label"].astype(int).to_numpy()


def make_vectorizer(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def metric_values(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def save_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=["actual_negative", "actual_positive"],
                  columns=["predicted_negative", "predicted_positive"]).to_csv(
                      RESULTS_DIR / "confusion_matrix.csv"
                  )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        xlabel="Predicted sentiment",
        ylabel="Actual sentiment",
        title="Confusion matrix — best-F1 TF-IDF model",
    )
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_threshold_plot(rows: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    thresholds = [row["threshold"] for row in rows]
    ax.plot(thresholds, [row["precision"] for row in rows], marker="o", label="Precision")
    ax.plot(thresholds, [row["recall"] for row in rows], marker="o", label="Recall")
    ax.set(
        xlabel="Positive-class decision threshold",
        ylabel="Score",
        title="Precision and recall by decision threshold",
        xticks=list(THRESHOLDS),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "threshold_precision_recall.png", dpi=160)
    plt.close(fig)


def summarize_review(text: str) -> str:
    """Describe a borderline review without reproducing its source text."""
    normalized = text.lower()
    if "love-hate" in normalized or ("fantastic" in normalized and "crap" in normalized):
        return (
            "The reviewer describes mixed feelings about musicals, contrasting "
            "strongly liked and disliked examples."
        )
    if "love horrible movies" in normalized:
        return (
            "The reviewer humorously presents the film as intentionally bad and "
            "uses an ironic claim of enjoying that kind of movie."
        )
    if "nothing really unpredictable" in normalized and "solid" in normalized:
        return (
            "The reviewer criticizes the film's predictability but balances that "
            "with praise for its overall execution."
        )
    if "bunch of sorority girls" in normalized and "mausoleum" in normalized:
        return (
            "The reviewer introduces a horror-film premise involving sorority "
            "pledges and a mausoleum, with little direct evaluation in the opening."
        )
    if "love letter" in normalized and "romantic comedy" in normalized:
        return (
            "The reviewer introduces a romantic-comedy plot and its setting, "
            "with little direct evaluation in the opening."
        )
    return "The review's opening provides context with limited explicit sentiment."


def analyze_borderline_language(text: str) -> str:
    normalized = text.lower()
    if "love-hate" in normalized or ("fantastic" in normalized and "crap" in normalized):
        return (
            "The writer explicitly describes a love-hate response and contrasts "
            "strong praise with strong criticism. A single positive/negative label "
            "compresses this mixed evaluation, and isolated words such as “love” "
            "or “fantastic” can pull in opposite directions from the overall verdict."
        )
    if "love horrible movies" in normalized:
        return (
            "The phrase “love horrible movies” is ironic: “love” is not praise for "
            "the film. The review also frames the movie as intentionally bad, so a "
            "bag-of-words model can mistake a locally positive word for the "
            "reviewer's overall sentiment."
        )
    if "nothing really unpredictable" in normalized and "solid" in normalized:
        return (
            "A criticism of predictability is immediately balanced by repeated "
            "praise (“solid”). This contrast and the understated wording can make "
            "the aggregate unigram and bigram evidence nearly cancel."
        )
    if "bunch of sorority girls" in normalized and "mausoleum" in normalized:
        return (
            "The opening is plot description rather than an explicit opinion. "
            "Genre words such as “creepy” and descriptions of the premise can "
            "dominate the opening excerpt without stating whether the reviewer "
            "liked the film."
        )
    if "love letter" in normalized and "romantic comedy" in normalized:
        return (
            "The excerpt mainly identifies the film and summarizes its setup, "
            "with little direct evaluative language. Plot and genre vocabulary "
            "may therefore outweigh the review's later judgment."
        )
    cues = []
    if any(term in normalized for term in ("not ", "never ", "no ")):
        cues.append("negation may reverse the polarity of nearby sentiment terms")
    if any(term in normalized for term in ("but ", "although ", "however ")):
        cues.append("contrast markers may separate praise from criticism")
    if not cues:
        cues.append("the excerpt is dominated by plot description or weak evaluative language")
    return "The review is borderline; " + " and ".join(cues) + "."


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data()
    train_texts, y_train = read_split("train")
    test_texts, y_test = read_split("test")

    specifications = {
        "Unigram TF-IDF": (1, 1),
        "Unigram + bigram TF-IDF": (1, 2),
    }
    model_results: list[dict[str, float | str]] = []
    model_probabilities: dict[str, np.ndarray] = {}
    for name, ngram_range in specifications.items():
        print(f"Fitting {name}...")
        vectorizer = make_vectorizer(ngram_range)
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)
        model = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=42)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        model_probabilities[name] = probabilities
        metrics = metric_values(y_test, probabilities, 0.5)
        model_results.append({"model": name, **metrics})
        print(f"  features={x_train.shape[1]:,}; F1={metrics['f1']:.4f}")

    comparison = pd.DataFrame(model_results).sort_values("f1", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_probabilities = model_probabilities[best_name]
    default_predictions = (best_probabilities >= 0.5).astype(int)
    save_confusion_matrix(y_test, default_predictions)

    threshold_rows = []
    for threshold in THRESHOLDS:
        threshold_rows.append(
            {"threshold": threshold, **metric_values(y_test, best_probabilities, threshold)}
        )
    pd.DataFrame(threshold_rows).to_csv(RESULTS_DIR / "threshold_metrics.csv", index=False)
    save_threshold_plot(threshold_rows)

    borderline_indices = np.argsort(np.abs(best_probabilities - 0.5))[:5]
    borderline_rows = []
    for index in borderline_indices:
        probability = float(best_probabilities[index])
        prediction = int(probability >= 0.5)
        borderline_rows.append(
            {
                "test_row": int(index),
                "actual_sentiment": "positive" if y_test[index] else "negative",
                "predicted_sentiment": "positive" if prediction else "negative",
                "positive_probability": probability,
                "review_summary": summarize_review(test_texts[index]),
            }
        )
    borderline = pd.DataFrame(borderline_rows)
    borderline.to_csv(RESULTS_DIR / "borderline_examples.csv", index=False)

    table = comparison.to_markdown(index=False, floatfmt=".4f")
    thresholds_table = pd.DataFrame(threshold_rows).to_markdown(index=False, floatfmt=".4f")
    borderline_table = borderline.to_markdown(index=False, floatfmt=".4f")
    delta_f1 = float(comparison.iloc[0]["f1"] - comparison.iloc[-1]["f1"])
    bigram_increased = str(comparison.iloc[0]["model"]) == "Unigram + bigram TF-IDF"
    if bigram_increased:
        comparison_note = (
            f"Unigram-plus-bigram TF-IDF had the higher test F1 by {delta_f1:.4f}; "
            "in this split, adding bigrams improved classification."
        )
    else:
        comparison_note = (
            f"Unigram TF-IDF had the higher test F1 by {delta_f1:.4f}; "
            "adding bigrams did not improve classification in this split."
        )

    borderline_lines = []
    for row in borderline_rows:
        borderline_lines.extend(
            [
                f"- Test row {row['test_row']}: actual **{row['actual_sentiment']}**, "
                f"predicted **{row['predicted_sentiment']}**, positive probability "
                f"**{row['positive_probability']:.4f}**.",
                f"  Review summary: {row['review_summary']}",
                f"  Analysis: {analyze_borderline_language(test_texts[row['test_row']])}",
            ]
        )

    report = f"""# IMDb Logistic Regression Sentiment Analysis

## Dataset and method

This experiment uses the official IMDb Large Movie Review Dataset split
(25,000 training and 25,000 test reviews), matching the split used in the
previous IMDb assignments. TF-IDF vocabularies and inverse-document-frequency
weights are fitted on training text only. Both Logistic Regression models use
C=1; the unigram model uses (1, 1) n-grams and the second uses (1, 2).

## (a)–(d) Model results

Precision, recall, and F1 are for the positive class.

{table}

## (e) Confusion matrix

The matrix and plot are for **{best_name}**, the higher-F1 model. Rows are
actual negative/positive, columns predicted negative/positive:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | {confusion_matrix(y_test, default_predictions, labels=[0, 1])[0, 0]:,} | {confusion_matrix(y_test, default_predictions, labels=[0, 1])[0, 1]:,} |
| Actual positive | {confusion_matrix(y_test, default_predictions, labels=[0, 1])[1, 0]:,} | {confusion_matrix(y_test, default_predictions, labels=[0, 1])[1, 1]:,} |

## (f) Feature comparison

{comparison_note} Bigrams can capture short expressions and local word order,
but also increase feature dimensionality and can split evidence across more
features. The result is specific to this dataset split and configuration.

## (g) Threshold results

The thresholds use positive-class probabilities from **{best_name}**.

{thresholds_table}

The plot `threshold_precision_recall.png` shows how increasing the threshold
generally trades recall for precision: more evidence is required to predict
positive, so fewer borderline reviews are labeled positive.

## (i)–(j) Five borderline test reviews

These are the five test examples whose positive probabilities are closest to
0.5. Review text is paraphrased rather than reproduced.

{borderline_table}

### Linguistic analysis

{chr(10).join(borderline_lines)}

Because these probabilities are near the 0.5 decision boundary, the classifier
has little net evidence for either class. Mixed positive and negative language,
negation (for example, positive terms in a negative clause), hedging, contrast
words, plot summaries, and sentiment that depends on context can all make
bag-of-words TF-IDF difficult to interpret. A short excerpt alone may not
capture the review's overall stance; the predicted probability is a model
confidence score, not a guarantee of correctness.
"""
    (RESULTS_DIR / "assignment_report.md").write_text(report, encoding="utf-8")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
