from sklearn import preprocessing,tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import csv

def main():
    train = readCsv("train.csv")
    train = rmIdCol(train)

    sample,target = splitTargetAndSample(train)

    sample = preProcessing(sample)
    sample = imputer(sample)

    print "rndForest"
    clf1 = rndForest(sample, target)
    print "adaboost"
    clf2 =adaboost(sample, target)
    print "gaussianbayes"
    clf3 =gaussianbayes(sample, target)
    print "knn"
    clf4 =knn(sample, target)


    mclf = tree.DecisionTreeClassifier(max_depth=10)
    mclf.fit(sample, target)

    sclf = StackingClassifier(classifiers=[clf1,clf3, clf2,clf4], meta_classifier=mclf)
    sclf = sclf.fit(sample, target)
    print "sclf"
    validation(sclf, sample, target)





    # test = readCsv("test.csv")
    # test = filterSparseData(test)
    # test = rmIdCol(test)
    # test = preProcessing(test)


    # print clf.predict(test)


def validation(clf, sample, target):
    scores = cross_val_score(clf, sample, target, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def knn(sample, target):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(sample, target)
    validation(clf, sample, target)
    return clf


def gaussianbayes(sample, target):
    gnb = GaussianNB()
    gnb.fit(sample, target)
    validation(gnb, sample, target)
    return gnb

def rndForest(sample, target):
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(sample, target)
    validation(clf, sample, target)
    return clf

def adaboost(sample, target):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), n_estimators=100)
    clf = clf.fit(sample, target)
    validation(clf, sample, target)
    return clf



def imputer(data):
    imp = Imputer(missing_values='NaN',strategy="most_frequent")
    odata = filterSparseData(data)
    imp.fit(odata)
    ndata = imp.transform(data)
    return ndata



def readCsv(file):
    data = []
    with open(file, 'rb') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
         for row in spamreader:
             data.append(row)

    return data[1:]


def filterSparseData(data):
    nData =[]
    for r in data:
        if r[3] != "":
            nData.append(r)
    return nData


def rmIdCol(data):
    sample =[]
    for r in data:
        sample.append(r[1:])
    return sample

def splitTargetAndSample(data):
    target =[]
    sample =[]
    for r in data:
        target.append(int(r[0]))
        sample.append(r[1:])
    return (sample,target)


def preProcessing(data):

    def selectFeatures(cols, data):
        nData =[]
        for r in data:
            row = []
            for col in cols:
                row.append(r[col])
            nData.append(row)
        return nData

    data = selectFeatures([0,2,3,4,5], data)


    def convertToNum(data):
        nData =[]
        for r in data:
            #pclass
            r[0] = int(r[0])

            # sex to male
            if r[1] == "male":
                r[1] = 1
            elif r[1] == "female":
                r[1] = 0
            else:
                r[1] = np.nan


            #age
            if r[2] != "" and float(r[2]) > 0:
                r[2] =float(r[2])
            else:
                r[2] = np.nan

            #sipsp
            r[3] = int(r[3])

            #parch
            r[4] = int(r[4])

            nData.append(r)
        return nData


    data = convertToNum(data)


    return data


if __name__ == '__main__':
    main()
