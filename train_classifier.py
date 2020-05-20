from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import numpy as np
import os
import string
import sys
import random
import re

random.seed(1991)
np.random.seed(1991)

def load_dataset(path):
    texts, labels = [], []
    with open(path) as handle:
        for new_line in handle:
            new_line = new_line.strip().split()

            toks = new_line[1:]
            fixed_toks = []
            for tok in toks:
                if re.match(r'r.*h.*n.*g.*a', tok):
                    fixed_toks.append('rohingya')
                else:
                    fixed_toks.append(tok)

            label = int(new_line[0])
            text = ' '.join(fixed_toks)

            if label != 2:
                texts.append(text)
                labels.append(label)

    return texts, labels

def build_pipeline():
    count_vect = CountVectorizer(ngram_range=(1, 3), binary=True)
    tfidf_transformer = TfidfTransformer()
    clf = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=1000, tol=None)
    calibrated_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    clf = Pipeline([('count', count_vect), ('clf', calibrated_clf)])

    return clf

def shuffle_data(texts, labels):
    combined = list(zip(texts, labels))
    random.shuffle(combined)

    texts[:], labels[:] = zip(*combined)

    return texts, labels

def train_model(clf, train_text, train_labels):
    clf = clf.fit(train_text, train_labels)

    return clf

def compute_metrics(clf, test_text, Y_test):    
    test_pred = clf.predict(test_text)

    test_probs = clf.predict_proba(test_text)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    y_test_arr = np.zeros((len(Y_test), n_classes))
    y_test_arr[np.arange(len(Y_test)), np.array(Y_test)] = 1
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_arr[:, i], test_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_arr.ravel(), test_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    cf = metrics.confusion_matrix(test_pred, Y_test)

    zeros_in_gt = np.where(np.array(Y_test)==0)
    for z in zeros_in_gt[0]:
        if test_pred[z] == 1:
            print(test_text[z])


    return roc_auc, cf, np.mean(test_pred == Y_test)


if __name__ == '__main__':
    dataset = sys.argv[1]
    dest = sys.argv[2]

    
    p_scores = []
    r_scores = []
    f1_scores = []
    acc_scores = []

    roc0 = []
    roc1 = []
    roc_micro = []
    
    for run in range(100):
        print('Run:', run)
        texts, labels = load_dataset(dataset)

        clf = build_pipeline()

        texts, labels = shuffle_data(texts, labels)

        n_train = int(0.9 * len(texts))

        train_text = texts[:n_train]
        test_text = texts[n_train:]

        train_labels = labels[:n_train]
        test_labels = labels[n_train:]    

        clf = train_model(clf, train_text, train_labels)
        roc_score, cf_mat, acc = compute_metrics(clf, test_text, test_labels)
        
        roc0.append(roc_score[0])
        roc1.append(roc_score[1])
        roc_micro.append(roc_score['micro'])
        
        p_num = cf_mat[1,1]
        p_den = cf_mat[1,0] + cf_mat[1,1]

        r_num = cf_mat[1,1]
        r_den = cf_mat[0,1] + cf_mat[1,1]

        #if p_den == 0:
        #    continue

        p = float(p_num) / p_den
        r = float(r_num) / r_den

        #if p + r == 0:
        #    continue

        f1 = 2 * (p * r) / (p + r)

        print('Metrics')
        print('Roc', roc_score)
        print('CF', cf_mat)
        print('Precision', p)
        print('Recall', p)
        print('Running precision', np.average(p_scores))
        print('Running recall', np.average(r_scores))
        print('Running acc', np.average(acc_scores))
        
        p_scores.append(p)
        r_scores.append(r)
        f1_scores.append(f1)
        acc_scores.append(acc)


        joblib.dump(clf, dest)
        

    print('Precision')
    print(np.mean(p_scores))
    print(np.std(p_scores))
    print('Recall')
    print(np.mean(r_scores))
    print(np.std(r_scores))
    print('F1')
    print(np.mean(f1_scores))
    print(np.std(f1_scores))
    print('Acc')
    print(np.mean(acc_scores))
    print(np.std(acc_scores))
    print('ROC_AUC0')
    print(np.mean(roc0))
    print(np.std(roc0))
    print('ROC_AUC1')
    print(np.mean(roc1))
    print(np.std(roc1))
    print('ROC_AUC_MICRO')
    print(np.mean(roc_micro))
    print(np.std(roc_micro))
