# find_duplicates.py
Find duplicates in affiliations dataset using XGBoost binary classifier from https://github.com/vkoe/binary-classifiers

Still in early stages -- so far:
    - basic model trained on jarowinkler score as feature

Next steps:
    - use fastText to calculate vector distance for each pair of names
    - train model with both jarowinkler score and fastText vector norms
    - (clean up all around)

