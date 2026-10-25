# Movie Success Patterns: Genre, Cast, Crew, Keywords, and Release Month

**Course project report**  
**Prepared:** 24 October 2026  
**Team members:** Add names and roll numbers  
**Topic uniqueness:** Confirm with the instructor/class list before submission  
**Dataset:** `TMDB  IMDB Movies Dataset.csv`

## Abstract

This project studies whether recurring combinations of movie genres, directors, lead cast, plot keywords, and release months are associated with high IMDb ratings or stronger reported revenue relative to budget. It converts each title into a typed transaction, mines frequent itemsets and outcome rules, compares eight mining approaches, and evaluates Decision Tree and Bernoulli Naive Bayes classifiers.

After deduplicating the supplied 443,147-row file by TMDB ID, 409,984 released titles had usable release dates. The rating analysis includes 177,269 titles with at least 100 IMDb votes; 22.6% meet the high-rating definition. The financial analysis has 10,119 released titles with positive budget and revenue; 65.1% have reported revenue above reported budget. Documentary-related combinations lead the high-rating lift table. In the lower-support people-focused analysis, Joseph Barbera with comedy has a lift of 3.63, with 91 supporting titles. For the revenue-to-budget proxy, horror + thriller + sequel has the highest reported lift in the displayed rules at 1.48, with 52 supporting titles.

The algorithms returned the same 1,990 frequent itemsets for the selected database and threshold where full frequent-set outputs were expected. Vertical ECLAT was fastest in this run, at approximately 0.05 seconds; these runtimes describe this reference implementation and computing environment only. Bernoulli Naive Bayes had the strongest PR-AUC for both classifier tasks, though the box-office results are based on a much smaller, selectively complete financial subset.

## 1. Problem statement and objectives

The project asks:

> Which combinations of genre, director, lead cast, keywords, and release month occur more often among successful movies than the corresponding outcome base rate?

The objectives are to:

1. Prepare the merged TMDB/IMDb movie metadata for transaction mining.
2. Compare Apriori and FP-Growth with hash-based Apriori, a vertical TID-bitmap miner, partitioned Apriori, a DIC checkpoint variant, closed-set mining, and maximal-set mining.
3. Calculate support, confidence, and lift for rules whose consequent is a success label.
4. Compare a Decision Tree and Bernoulli Naive Bayes classifier using only pre-release metadata features.
5. Discuss how data quality, outcome definitions, support thresholds, and missing fields limit the conclusions.

## 2. Dataset and preparation

The supplied file is the [TMDB + IMDb Merged Movies Dataset on Kaggle](https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset/data). Its columns include TMDB and IMDb identifiers and ratings, release date, genres, keywords, directors, cast, budget, revenue, and vote counts.

### 2.1 Cleaning decisions

| Data stage | Rows |
|---|---:|
| Source CSV | 443,147 |
| Repeated TMDB-ID rows removed | 13,448 |
| Unique TMDB-ID rows retained | 429,699 |
| Released titles with a valid release date on/before the run date | 409,984 |
| Rating analysis: released titles with at least 100 IMDb votes | 177,269 |
| Financial analysis: released titles with positive budget and revenue | 10,119 |

For duplicate TMDB IDs, the retained row is the one with the most complete required metadata; ties are resolved by IMDb vote count and then TMDB vote count. The data contains 16,474 excess rows with repeated IMDb IDs as well, so cross-source identifier conflicts remain a limitation even after keeping one row per TMDB ID.

Among the released titles, keywords are missing for 59.2%, genres for 15.2%, cast for 14.4%, and directors for 2.2%. Missing values are left out of the transaction rather than assigned a synthetic item. This treats an absent keyword as unknown/unsupplied, not as evidence that the movie has no keywords.

Only rows marked `Released` with a valid release date on or before the run date are used for retrospective outcomes. The source has no separate theatrical-release flag, so records tagged as `TV Movie` can remain in the rating analysis. This taxonomy should be considered when interpreting rules involving that genre.

### 2.2 Transaction representation

Each title is represented by a set of typed items:

- Genres: `GENRE::drama`
- Director: `DIRECTOR::christopher nolan`
- First three credited cast names: `ACTOR::...`
- First eight listed keywords: `KEYWORD::...`
- Release month: `MONTH::07`
- Exactly one outcome item, such as `OUTCOME::HighRating`

Type prefixes prevent, for example, an actor name from being confused with a keyword. Limiting cast and keywords makes the basket size and search vocabulary manageable. For the main mining comparison, the vocabulary contains all observed genres and months, plus up to 20 directors, 30 actors, and 40 keywords ranked by frequency. That yields 123 items including the outcome categories.

## 3. Success labels and association measures

### High IMDb rating

A title is labeled high-rated when `averageRating >= 7.0`, after requiring at least 100 IMDb votes. This leaves 177,269 titles, of which 40,054 (22.6%) are high-rated.

### Revenue-to-budget outcome

A title is labeled positive when its reported revenue exceeds its reported budget, after requiring both values to be positive. This leaves 10,119 titles, of which 6,586 (65.1%) meet the rule. It is called **revenue above budget** throughout this report; it is only a proxy and does not establish profit because it omits marketing, distribution, taxes, inflation, and possible source errors.

### Rule measures

For antecedent (X) and outcome (H):

- **Support:** (P(X \cup H)), the share of eligible titles containing both.
- **Confidence:** (P(H \mid X)), the share of titles with (X) that also meet the outcome.
- **Lift:** (P(H \mid X) / P(H)), confidence divided by the outcome base rate.

Lift above 1 indicates that the outcome is more common among titles with the pattern than in the eligible dataset. It is an association, not a causal claim or a promise of success.

## 4. Frequent-itemset methods

All methods in the main comparison use the same 177,269 rating-eligible transactions, a 0.15% minimum support threshold (266 titles), and maximum itemset length 4.

| Method | Implementation in this project | Output in this run |
|---|---|---:|
| Apriori | Level-wise candidate generation and subset pruning | 1,990 frequent sets |
| FP-Growth | FP-tree and conditional pattern bases | 1,990 frequent sets |
| Hash Apriori | Hash buckets prune some 2-item candidates; exact support verifies survivors | 1,990 frequent sets |
| Vertical ECLAT | Integer TID bitmaps intersected to count support | 1,990 frequent sets |
| Partitioned Apriori | Local frequent sets are unioned and globally verified | 1,990 frequent sets |
| DIC checkpoint variant | Candidates are added at support checkpoints and verified on the full database | 1,990 frequent sets |
| A-Close closure filtering | Closed sets are selected from the Apriori support map | 1,989 closed sets |
| Pincer bounded maximal mode | Maximal frequent sets and a capped MFCS border are tracked | 822 maximal sets |

The DIC implementation is a course-friendly checkpoint variant: it uses support observed on successive data prefixes to introduce candidates, then verifies exact full-database support. It does not implement the full textbook circular-scan state machine.

The A-Close implementation applies equal-support closure filtering to Apriori's frequent sets. Its reported runtime includes Apriori plus the closure pass; its independent candidate count is not reported. It demonstrates the closed-itemset idea but is not the optimized A-Close search procedure.

The Pincer implementation is also explicitly bounded and educational. It tracks a maximal-frequent-set collection and a capped MFCS border while retaining the level-wise candidate frontier for completeness up to length 4. Its maximal-set output is not directly comparable to the all-frequent-set outputs.

### 4.1 Measured algorithm results

Runtimes below were measured in one local execution. They vary with hardware, Python version, memory, and implementation details.

| Algorithm | Output mode | Results | Candidate/node count | Runtime (seconds) |
|---|---|---:|---:|---:|
| Vertical ECLAT | All frequent sets | 1,990 | 1,990 intersections | 0.050 |
| Apriori | All frequent sets | 1,990 | 7,104 | 0.065 |
| A-Close closure filtering | Closed sets | 1,989 | — | 0.067 end-to-end with Apriori |
| Hash Apriori | All frequent sets | 1,990 | 5,387 | 0.293 |
| Partitioned Apriori | All frequent sets | 1,990 | 4,067 union candidates | 0.396 |
| DIC checkpoint variant | All frequent sets, verified | 1,990 | 11,881 discovered | 0.428 |
| FP-Growth | All frequent sets | 1,990 | 1,990 mined nodes | 0.819 |
| Pincer bounded maximal mode | Maximal sets | 822 | 7,104 level-wise candidates | 4.793 |

The agreement among Apriori, FP-Growth, Hash Apriori, ECLAT, partitioned Apriori, and the DIC checkpoint variant is a useful consistency observation for this run. A-Close and Pincer return intentionally reduced outputs. The lower runtime for ECLAT reflects this implementation's compact bitmap intersections and should not be generalized to every transaction database.

## 5. Association-rule results

### 5.1 High-rating rules

The main run uses 0.15% minimum support and 22.6% high-rating base rate. The top displayed rules are broad documentary combinations:

| Pattern | Support count | Confidence | Base rate | Lift |
|---|---:|---:|---:|---:|
| Documentary + Music | 1,308 | 77.4% | 22.6% | 3.42 |
| Documentary + Biography keyword | 485 | 73.2% | 22.6% | 3.24 |
| Documentary + TV Movie | 354 | 67.8% | 22.6% | 3.00 |
| Documentary + Sports keyword | 278 | 66.5% | 22.6% | 2.94 |
| Documentary + History | 620 | 66.0% | 22.6% | 2.92 |
| Documentary + November release | 1,084 | 63.0% | 22.6% | 2.79 |

The lower-support people analysis uses 0.03% support (54 titles) and maximum length 3 to surface cast and crew patterns. It reports examples such as:

| Pattern | Support count | Confidence | Base rate | Lift |
|---|---:|---:|---:|---:|
| Joseph Barbera + Comedy | 91 | 82.0% | 22.6% | 3.63 |
| William Hanna + Comedy | 91 | 81.2% | 22.6% | 3.60 |
| Joseph Barbera + William Hanna | 108 | 77.1% | 22.6% | 3.41 |
| Larry Fine + Moe Howard | 147 | 72.4% | 22.6% | 3.20 |

These people-specific rules are exploratory because they use a lower support threshold. Their counts are shown so the high lifts are not read without context. The recurring animation/comedy and performer-pair patterns may reflect catalog structure and repeated credits as well as audience response.

### 5.2 Revenue-above-budget rules

Only 10,119 titles have positive reported revenue and budget. The positive base rate is 65.1%, so lifts are compared to that relatively high baseline.

| Pattern | Support count | Confidence | Base rate | Lift |
|---|---:|---:|---:|---:|
| Horror + Thriller + Sequel | 52 | 96.3% | 65.1% | 1.48 |
| Thriller + Sequel | 98 | 94.2% | 65.1% | 1.45 |
| Horror + Sequel | 87 | 92.6% | 65.1% | 1.42 |
| Adventure + Science Fiction + June | 53 | 91.4% | 65.1% | 1.40 |
| Action + Sequel | 121 | 91.0% | 65.1% | 1.40 |

These are patterns in the complete-financial-record subset, not estimates for all titles. Because budget and revenue coverage is selective, the eligible movies may differ from the full catalog.

## 6. Classification

The classifiers predict the rating and revenue-to-budget labels in separate models. Predictors are typed genres, director, first three credited cast names, first eight keywords, and release month. Ratings, vote counts, popularity, revenue, and target fields are excluded from the predictors. The model input is a sparse binary matrix with up to 3,000 frequent features. A stratified 80/20 random holdout uses seed 42.

- **Decision Tree:** Gini criterion, maximum depth 12, minimum leaf size 20, balanced class weights.
- **Bernoulli Naive Bayes:** binary presence features and additive smoothing `alpha=1`.

| Task | Model | Test rows | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High rating | Decision Tree | 35,454 | 0.629 | 0.663 | 0.347 | 0.726 | 0.469 | 0.737 | 0.458 |
| High rating | Bernoulli Naive Bayes | 35,454 | 0.789 | 0.659 | 0.544 | 0.421 | 0.474 | 0.764 | 0.495 |
| Revenue > budget | Decision Tree | 2,024 | 0.570 | 0.566 | 0.708 | 0.577 | 0.636 | 0.598 | 0.723 |
| Revenue > budget | Bernoulli Naive Bayes | 2,024 | 0.633 | 0.586 | 0.708 | 0.743 | 0.725 | 0.642 | 0.757 |

Bernoulli Naive Bayes has the strongest PR-AUC in both tasks, while the Decision Tree has higher rating recall. The box-office PR-AUC should be compared with its 65.1% positive baseline and interpreted cautiously because only 10,119 titles enter that task. This random split measures within-dataset holdout performance; it does not show how well models generalize to future releases.

## 7. Conclusions

The merged dataset is a strong fit for this course project because it combines the attributes needed for itemsets and classification in one file and offers enough rows for a meaningful algorithm comparison. It also requires real data-quality work: IDs repeat, keywords are missing for a majority of released titles, and only a small fraction has both budget and revenue.

Within the chosen definitions, documentary genre combinations are strongly associated with high ratings. Lower-support people patterns surfaced repeated animation/comedy credits. The revenue-above-budget patterns are more modest in lift, with sequel-related combinations leading the displayed results. These are dataset-specific associations and should not be presented as causal factors.

For a final classroom presentation, emphasize the high-rating analysis as the primary result because it uses 177,269 eligible titles. Present the financial analysis as secondary, state its 10,119-title sample, and call its target revenue-above-budget rather than profitability. Include the support count beside each rule and clearly disclose the DIC, A-Close, and Pincer implementation scope described above.

## 8. Reproducibility and submission checklist

The executed notebook is `Movie_Success_Patterns_Project.ipynb`; algorithms are in `movie_mining.py`. Keep both beside the supplied CSV. Install the packages listed in `requirements.txt`, open the notebook, and run all cells from top to bottom. The notebook stores its outputs and figures.

Before submission:

- Add the team names and roll numbers.
- Confirm that no other team has the same approved topic.
- Check the Kaggle page's license and attribution details and follow source requirements for TMDB/IMDb data.
- Replace any report statements required by your instructor's exact definitions or algorithm expectations.
- Submit the notebook, source module, report, requirements file, and the CSV only if course rules permit redistribution.

## References

1. [TMDB + IMDb Merged Movies Dataset, Kaggle](https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset/data).
2. Agrawal, R. and Srikant, R. (1994). [Fast Algorithms for Mining Association Rules](https://www.vldb.org/conf/1994/P487.PDF). VLDB.
3. Han, J., Pei, J., and Yin, Y. (2000). [Mining Frequent Patterns without Candidate Generation](https://doi.org/10.1145/335191.335372). SIGMOD.
4. Brin, S., Motwani, R., Ullman, J. D., and Tsur, S. (1997). [Dynamic Itemset Counting and Implication Rules for Market Basket Data](https://doi.org/10.1145/253260.253325). SIGMOD.
5. Park, J. S., Chen, M.-S., and Yu, P. S. (1995). [An Effective Hash-Based Algorithm for Mining Association Rules](https://doi.org/10.1145/223784.223813). SIGMOD.
6. Savasere, A., Omiecinski, E., and Navathe, S. B. (1995). [An Efficient Algorithm for Mining Association Rules in Large Databases](https://www.vldb.org/dblp/db/conf/vldb/SavasereON95.html). VLDB.
7. Zaki, M. J. (2000). [Scalable Algorithms for Association Mining](https://doi.org/10.1109/69.846291). IEEE Transactions on Knowledge and Data Engineering.
8. Pasquier, N., Bastide, Y., Taouil, R., and Lakhal, L. (1999). [Discovering Frequent Closed Itemsets for Association Rules](https://doi.org/10.1007/3-540-49257-7_25). ICDT.
9. Lin, D.-I. and Kedem, Z. M. (1998). [Pincer-Search: A New Algorithm for Discovering the Maximum Frequent Set](https://www.sigmod.org/publications/dblp/db/conf/edbt/LinK98.html). EDBT.
10. [scikit-learn DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [BernoulliNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html).
11. [TMDB API attribution requirements](https://developer.themoviedb.org/docs/faq).
