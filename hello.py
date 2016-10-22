from sklearn import tree
import numpy as np

features = [[130, 0], [180, 1], [120, 0], [170, 1]]
labels = [0, 1, 0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[150, 0]])
