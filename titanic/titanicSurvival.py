from sklearn import preprocessing
import numpy as np
import csv

class Passenger:
    survived=False
    pclass=False
    sex=False
    age=False
    sibsp=False
    parch=False

    def __init__(self, csvRow):
        self.survived = csvRow[1]
        self.pclass= csvRow[2]
        self.sex= csvRow[4]
        self.age= csvRow[5]
        self.sibsp= csvRow[6]
        self.parch= csvRow[7]



def readTrain():
    train = []
    with open('train.csv', 'rb') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
         for row in spamreader:
             train.append(row)

    return train[1:]

def dataToPassenger(data):
    nData =[]
    for row in data:
        nData.append(Passenger(row))
    return nData



train = readTrain()
res = dataToPassenger(train)
print res


