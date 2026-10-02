"""ML ranking layer for the foundation matcher.

Two-stage pipeline
------------------
1. ``rules.py`` hard-filters the catalog and produces an auditable score.
2. This package re-ranks those survivors using a model trained on
   interaction logs (clicked / saved / purchased / rated).

The model never resurrects a filtered-out product. It only reorders
candidates the rules already approved.

First models
------------
- Pointwise logistic regression (interpretable baseline).
- Pointwise XGBoost for non-linear interactions such as dry × satin.
- Pairwise logistic ranking (RankSVM-style) when a session has both
  positives and negatives.
- ``xgbranker`` (LambdaMART-style) once usage volume supports group splits.

Training targets: ``clicked``, ``saved``, ``purchased``, ``rated`` (4+),
or composite ``relevance``.
"""
