# Movie Success Patterns project

This project mines frequent combinations of movie genres, directors, cast, keywords, and release month, then compares Decision Tree and Bernoulli Naive Bayes classifiers for high IMDb rating and reported revenue above budget.

## Files

- `Movie_Success_Patterns_Project.ipynb` — executed Jupyter notebook with tables, charts, algorithm comparisons, association rules, and classifier results.
- `movie_mining.py` — implementations of the frequent-itemset methods used by the notebook.
- `project_report.md` — detailed report with the measured findings and method limitations.
- `TMDB  IMDB Movies Dataset.csv` — supplied source dataset.
- `requirements.txt` — Python packages used by the notebook.

## Run the notebook

1. Keep the notebook, `movie_mining.py`, and CSV in the same folder.
2. Install Python 3.11 or newer and the packages in `requirements.txt`.
3. Start JupyterLab or Jupyter Notebook, open `Movie_Success_Patterns_Project.ipynb`, select the installed Python kernel, and choose **Run All**.

The notebook reads the CSV from its current folder. It can take a little while to load and process this large file. The notebook already contains saved outputs, so the tables and plots are visible before rerunning.

## Important interpretation notes

- “High rating” means IMDb average rating of at least 7.0 with at least 100 IMDb votes.
- “Revenue above budget” is a limited proxy, not a profit measure.
- DIC, A-Close, and Pincer are explicitly labeled educational variants in the notebook and report; see the implementation notes before describing them as textbook-optimized implementations.
- Add team details and confirm topic uniqueness before submission. Check dataset and source attribution terms before redistributing the CSV.
