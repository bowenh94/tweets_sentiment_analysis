from sklearn.feature_selection import f_classif,chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm
import numpy as np
import arff
import scipy

#load raw trainning data
raw_train_t = pd.read_table('~/2017S1-KTproj2-data/train-tweets.txt',header = None,usecols = [0,1])
raw_train_t.columns = ['ID','Tweet']
raw_train_l = pd.read_table('~/2017S1-KTproj2-data/train-labels.txt',header = None,usecols = [0,1])
raw_train_l.columns = ['ID','Label']
train_df = pd.merge(raw_train_t,raw_train_l,on='ID')

test_real = pd.read_table('~/2017S1-KTproj2-data/test-tweets.txt',header = None,usecols = [0,1])
test_real.columns = ['ID','Tweet']


from sklearn.feature_extraction.text import TfidfTransformer
cv = CountVectorizer(min_df = 0.00001,stop_words = 'english',analyzer='word',ngram_range=(1,2))
train_X_Miko = cv.fit_transform(train_df.Tweet)
test_X_Miko = cv.transform(test_real.Tweet)


tf_fea = TfidfTransformer()
train_X_TfFS = tf_fea.fit_transform(train_X_Miko)
test_X_TfFS = tf_fea.transform(test_X_Miko)


skb02 = SelectPercentile(score_func=chi2,percentile = 0.46)
train_X = skb02.fit_transform(train_X_TfFS,train_df.Label)
test_X = skb02.transform(test_X_TfFS)

svm = SVC(gamma=10, C=1.29)
y_pred = svm.fit(train_X,train_df.Label).predict(test_X)

actual = ["?" for _ in range(4924)]
error = ["" for _ in range(4924)]
prediction = ["" for _ in range(4924)]
r= {'actual':actual,'predicted':y_pred,'actual':actual,'error':error,'prediction':prediction,'id':test_real.ID}
result = pd.DataFrame(data = r)
result.to_csv("~/output_test.csv")
