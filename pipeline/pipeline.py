from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
#give accuracy_score() the labels of testing data, and the labels output by the classifier, and it will give you an accuracy score for the classifier
#accuracy_score() will give different output when called more than once - bc of "randomness in how train/test data is partitioned"
print accuracy_score(y_test, predictions)


#use different classifier
#these two lines are the only different lines than the code for doing the same thing with the decision tree - ie, different classifiers work the same way at a high level
from sklearn.neighbors import KNeighborsClassifier
nbr_clf = KNeighborsClassifier()

nbr_clf.fit(X_train, y_train)
nbr_predictions = nbr_clf.predict(X_test)
print accuracy_score(y_test, nbr_predictions)


'''
def classify(features):
    #logic
    return label

- in this function the logic is the part we want the classifier to learn
- use training data to adjust parameters of the model
'''
