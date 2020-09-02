"""
Train a simple bag-of-words classifier
for personal attacks using
the Wikipedia Talk Labels: Personal Attacks data set.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_PATH = "/Users/arjit/thesis/WOAH-2020-Shared-Exploration/WOAH-2020-Shared-Exploration/WOAHTaskDataset/"
ANNOTATED_COMMENTS_PATH = BASE_PATH + "attack_annotated_comments.tsv"
ANNOTATIONS_PATH = BASE_PATH + "attack_annotations.tsv"

def main():
    comments = pd.read_csv(ANNOTATED_COMMENTS_PATH, sep='\t', index_col = 0)
    annotations = pd.read_csv(ANNOTATIONS_PATH, sep='\t')

    print(len(annotations['rev_id'].unique())) # --> 115864

    # Label a comment as an attack if the majority of annotators did so
    labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
    comments['attack'] = labels


    # remove newline and tab tokens
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

    print(comments.query('attack')['comment'].head())

    train_comments = comments.query("split=='train'")
    test_comments = comments.query("split=='test'")

    clf = Pipeline([
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', LogisticRegression()),
    ])
    clf = clf.fit(train_comments['comment'], train_comments['attack'])
    auc = roc_auc_score(test_comments['attack'], clf.predict_proba(test_comments['comment'])[:, 1])
    print('Test ROC AUC: %.3f' % auc)

    print(clf.predict(['What\'s up, dude?', 'What\'s up, nigga?']))
    print(clf.predict(['Are you a nigga?']))
    print(clf.predict(['Are you a black man?', 'Are you as fat as a black man?', 'Are you a nigga?', 'Are you as fat as a black man?', 'Is this racist?']))
    print(clf.predict(['What do you call a movie with an all- Muslim cast? A box office bomb.', 'We shouldn’t lower our standards just to hire more women.',
                       'Kendrick just dropped the coldest video of all fucking time', ]))

if __name__ == "__main__":
    main()