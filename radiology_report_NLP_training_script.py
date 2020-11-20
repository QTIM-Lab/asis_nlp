'''
Stroke CT/MRI NLP model training script
'''

import os
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.sentiment.util import mark_negation
from nltk.stem.snowball import EnglishStemmer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# load spreadsheet containing radiology report free-text impressions and a binary label for acute or subacute ischemic stroke (ASIS)
# columns are named 'IMPRESSION' and 'ACUTE_INFARCT'
sample_data = pd.read_csv('radiology_report_data.csv')

'''
Helper functions for stemming and tokenize by sentences
'''

stemmer = EnglishStemmer()
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def report_excerpt_fun(report_data):
    # filter each report in dataframe for reg ex containing only sentences of interest
    # also stem words in the sentence
    report_excerpts = []
    for i in tqdm(range(len(report_data))):
        report = report_data.iloc[i]
        # regex identifies sentences containing infarct or ischem
        x = re.findall(r"([^.]*?(infarct|ischem)[^.]*\.)", report, flags=re.IGNORECASE) 
        report_excerpt = ' '.join([a[0] for a in x])
        report_excerpt = stemSentence(report_excerpt)
        report_excerpts.append(report_excerpt)
    return(report_excerpts)

# BoW vectorizer - N-gram

vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = lambda text: mark_negation(word_tokenize(text)), # note use of NLTK mark_negation 
                             ngram_range = (2,3), 
                             min_df = 0.01 # word shows up in at least N% of samples
                             )

# train using the annotated radiology report data

X = sample_data['IMPRESSION']
y = sample_data['ACUTE_INFARCT']

X_ = report_excerpt_fun(X)
X_v = vectorizer.fit_transform(X_)

# K-fold cross validation testing

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=88)
skf.get_n_splits(X_v, y)

cv_results = {'acc': [], 'prec': [], 'recall': [], 'f1': [], 'tn': [], 'fp': [], 'fn': [], 'tp': [], 'pred_pos':[], 'actual_pos':[]}

for train_index, test_index in tqdm(skf.split(X_v, y)):
    X_train, X_test = X_v[train_index], X_v[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    clf_AIS_rf = RandomForestClassifier(n_estimators = 100).fit(X_train, list(y_train))
    predicted = clf_AIS_rf.predict(X_test)

    test_labels = list(y_test)

    acc = accuracy_score(test_labels, predicted)
    prec = precision_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    f1 = f1_score(test_labels, predicted)

    tn, fp, fn, tp = confusion_matrix(test_labels, predicted).ravel()

    pred_pos = tp + fp
    actual_pos = tp + fn

    cv_results['acc'].append(acc)
    cv_results['prec'].append(prec)
    cv_results['recall'].append(recall)
    cv_results['f1'].append(f1)

    cv_results['tn'].append(tn)
    cv_results['fp'].append(fp)
    cv_results['fn'].append(fn)
    cv_results['tp'].append(tp)

    cv_results['pred_pos'].append(pred_pos)
    cv_results['actual_pos'].append(actual_pos)


