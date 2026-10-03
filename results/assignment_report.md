# IMDb Logistic Regression Sentiment Analysis

## Dataset and method

This experiment uses the official IMDb Large Movie Review Dataset split
(25,000 training and 25,000 test reviews), matching the split used in the
previous IMDb assignments. TF-IDF vocabularies and inverse-document-frequency
weights are fitted on training text only. Both Logistic Regression models use
C=1; the unigram model uses (1, 1) n-grams and the second uses (1, 2).

## (a)–(d) Model results

Precision, recall, and F1 are for the positive class.

| model                   |   accuracy |   precision |   recall |     f1 |
|:------------------------|-----------:|------------:|---------:|-------:|
| Unigram + bigram TF-IDF |     0.8956 |      0.8915 |   0.9009 | 0.8961 |
| Unigram TF-IDF          |     0.8884 |      0.8878 |   0.8890 | 0.8884 |

## (e) Confusion matrix

The matrix and plot are for **Unigram + bigram TF-IDF**, the higher-F1 model. Rows are
actual negative/positive, columns predicted negative/positive:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 11,129 | 1,371 |
| Actual positive | 1,239 | 11,261 |

## (f) Feature comparison

Unigram-plus-bigram TF-IDF had the higher test F1 by 0.0077; in this split, adding bigrams improved classification. Bigrams can capture short expressions and local word order,
but also increase feature dimensionality and can split evidence across more
features. The result is specific to this dataset split and configuration.

## (g) Threshold results

The thresholds use positive-class probabilities from **Unigram + bigram TF-IDF**.

|   threshold |   accuracy |   precision |   recall |     f1 |
|------------:|-----------:|------------:|---------:|-------:|
|      0.3000 |     0.8168 |      0.7377 |   0.9832 | 0.8429 |
|      0.5000 |     0.8956 |      0.8915 |   0.9009 | 0.8961 |
|      0.7000 |     0.8106 |      0.9701 |   0.6410 | 0.7720 |

The plot `threshold_precision_recall.png` shows how increasing the threshold
generally trades recall for precision: more evidence is required to predict
positive, so fewer borderline reviews are labeled positive.

## (i)–(j) Five borderline test reviews

These are the five test examples whose positive probabilities are closest to
0.5. Review text is paraphrased rather than reproduced.

|   test_row | actual_sentiment   | predicted_sentiment   |   positive_probability | review_summary                                                                                                                          |
|-----------:|:-------------------|:----------------------|-----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
|        580 | negative           | negative              |                 0.5000 | The reviewer introduces a romantic-comedy plot and its setting, with little direct evaluation in the opening.                           |
|      14435 | positive           | negative              |                 0.4999 | The reviewer criticizes the film's predictability but balances that with praise for its overall execution.                              |
|       6627 | negative           | positive              |                 0.5001 | The reviewer humorously presents the film as intentionally bad and uses an ironic claim of enjoying that kind of movie.                 |
|      24864 | positive           | positive              |                 0.5001 | The reviewer introduces a horror-film premise involving sorority pledges and a mausoleum, with little direct evaluation in the opening. |
|        188 | negative           | positive              |                 0.5002 | The reviewer describes mixed feelings about musicals, contrasting strongly liked and disliked examples.                                 |

### Linguistic analysis

- Test row 580: actual **negative**, predicted **negative**, positive probability **0.5000**.
  Review summary: The reviewer introduces a romantic-comedy plot and its setting, with little direct evaluation in the opening.
  Analysis: The excerpt mainly identifies the film and summarizes its setup, with little direct evaluative language. Plot and genre vocabulary may therefore outweigh the review's later judgment.
- Test row 14435: actual **positive**, predicted **negative**, positive probability **0.4999**.
  Review summary: The reviewer criticizes the film's predictability but balances that with praise for its overall execution.
  Analysis: A criticism of predictability is immediately balanced by repeated praise (“solid”). This contrast and the understated wording can make the aggregate unigram and bigram evidence nearly cancel.
- Test row 6627: actual **negative**, predicted **positive**, positive probability **0.5001**.
  Review summary: The reviewer humorously presents the film as intentionally bad and uses an ironic claim of enjoying that kind of movie.
  Analysis: The phrase “love horrible movies” is ironic: “love” is not praise for the film. The review also frames the movie as intentionally bad, so a bag-of-words model can mistake a locally positive word for the reviewer's overall sentiment.
- Test row 24864: actual **positive**, predicted **positive**, positive probability **0.5001**.
  Review summary: The reviewer introduces a horror-film premise involving sorority pledges and a mausoleum, with little direct evaluation in the opening.
  Analysis: The opening is plot description rather than an explicit opinion. Genre words such as “creepy” and descriptions of the premise can dominate the opening excerpt without stating whether the reviewer liked the film.
- Test row 188: actual **negative**, predicted **positive**, positive probability **0.5002**.
  Review summary: The reviewer describes mixed feelings about musicals, contrasting strongly liked and disliked examples.
  Analysis: The writer explicitly describes a love-hate response and contrasts strong praise with strong criticism. A single positive/negative label compresses this mixed evaluation, and isolated words such as “love” or “fantastic” can pull in opposite directions from the overall verdict.

Because these probabilities are near the 0.5 decision boundary, the classifier
has little net evidence for either class. Mixed positive and negative language,
negation (for example, positive terms in a negative clause), hedging, contrast
words, plot summaries, and sentiment that depends on context can all make
bag-of-words TF-IDF difficult to interpret. A short excerpt alone may not
capture the review's overall stance; the predicted probability is a model
confidence score, not a guarantee of correctness.
