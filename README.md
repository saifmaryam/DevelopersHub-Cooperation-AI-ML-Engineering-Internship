# AI/ML Engineering Internship — DevelopersHub Corporation

Hi, I'm Maryam — an MS Computer Science (Data Science), and this repo holds the work I did during my AI/ML Engineering internship at DevelopersHub Corporation.

Each notebook started from the same place — a clean Colab environment — and grew through actual debugging, dataset mirror swaps when Kaggle links went stale, and library version mismatches that needed fixing along the way. What's here is what actually ran, not a polished first draft.

---

## What's Inside

| # | Project | What It Does |
|---|---------|---------------|
| 1 | [Iris Dataset Exploration](./Task1_Iris_EDA.ipynb) | Full visual exploration of the classic Iris dataset — distributions, outliers, pairwise relationships |
| 2 | [Stock Price Forecasting](./Task2_Stock_Price_Prediction.ipynb) | Predicting AAPL's next-day closing price from 5 years of historical data |
| 3 | [Heart Disease Risk Classifier](./Task3_Heart_Disease_Prediction.ipynb) | Binary classification on patient health indicators |
| 5 | [Mental Health Support Bot](./Task5_Mental_Health_Chatbot.ipynb) | Fine-tuned DistilGPT2 for empathetic conversational responses |
| 6 | [House Price Estimator](./Task6_House_Price_Prediction.ipynb) | Regression on the Ames Housing dataset with engineered features |

## Task 1 — Exploring the Iris Dataset

Nothing fancy here on purpose — this was about getting comfortable reading a dataset properly before modeling anything. Loaded directly from seaborn, then walked through histograms, box plots for outliers, a correlation heatmap, and a full pair plot.

**What stood out:** petal length and petal width are doing almost all the work separating the three species — Setosa peels off cleanly, while Versicolor and Virginica overlap a bit more. Sepal width turned out to be the noisiest feature, with the most outliers concentrated in Setosa.

---

## Task 2 — Can You Predict Tomorrow's Stock Price?

Pulled 5 years of AAPL data via `yfinance`, engineered lag features (1, 2, 3, 5-day) and rolling statistics, then compared Linear Regression against Random Forest and Gradient Boosting — using a proper time-respecting train/test split, not a random shuffle (shuffling time series data quietly inflates your scores).

**What stood out:** yesterday's closing price alone explains most of tomorrow's. The fancier engineered features added only marginal lift over that one lag feature — a useful reminder that simple baselines are hard to beat in price prediction.

---

## Task 3 — Heart Disease Risk from Patient Data

This one had a real hiccup worth mentioning: the original Kaggle dataset (`ronitf/heart-disease-uci`) had been taken down mid-project, so I had to track down a working mirror and adjust the loading logic. Compared Logistic Regression, a Decision Tree, and Random Forest using ROC-AUC as the primary metric since the classes aren't perfectly balanced.

**What stood out:** max heart rate achieved and ST depression (a specific ECG measurement) were consistently the strongest predictors — both have known clinical relevance, which was reassuring to see reflected in the model's own feature importances rather than just trusting the numbers blindly.

---

## Task 5 — Teaching a Small Model to Respond with Empathy

Fine-tuned DistilGPT2 (82M parameters — small enough to train on a free Colab GPU) on emotionally-grounded conversational data, filtered down to anxiety, stress, loneliness, and related emotional contexts rather than the full general-purpose dataset.

**What stood out:** even a relatively tiny model picks up tone and conversational rhythm fast once fine-tuned — proof that you don't always need a massive model to get a noticeably warmer, more context-aware response than the base model produces out of the box.

---

## Task 6 — Estimating House Prices

Used the Ames Housing dataset and went past the obvious features (square footage, bedroom count) into engineered ones — total bathrooms, house age at time of sale, whether it had been remodeled, quality-weighted living area. Compared five regression models including Gradient Boosting and Extra Trees.

**What stood out:** overall quality rating outweighs raw square footage as a price driver, and neighborhood alone accounts for a large price swing — which lines up with how real estate actually gets priced in practice, not just by size.

---

## Tools & Stack

`Python` · `pandas` · `scikit-learn` · `PyTorch` · `HuggingFace Transformers` · `matplotlib` / `seaborn` · `yfinance` · Google Colab (T4 GPU for Task 5)

---

## Notes on Running These

- Tasks 3 and 6 originally pulled data via the Kaggle API — if a `kaggle.json` token isn't available, both notebooks have a GitHub-mirror fallback documented in the early cells.
- Task 5 needs a GPU runtime in Colab (Runtime → Change runtime type → T4 GPU) — training on CPU works but is significantly slower.
- All notebooks are self-contained — running top to bottom should reproduce every result and chart shown.

---

*MS Computer Science (Data Science), University of Gujrat — 2024–2026*
