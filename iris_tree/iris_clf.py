import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from IPython.display import Image


iris = load_iris()
test_idx = [0, 50, 100]

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
#.fit() is classifying alg
#after using .fit() with training data, the clf classifier can .predict()
clf.fit(train_data, train_target)

print 'training target: ', test_target
#predictor: give features, it will return labels
print 'training result: ', clf.predict(test_data)


with open('iris.dot', 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

'''
then run bash command

$ dot -Tpng iris.dot -o iris_tree.png

to generate .png of decision tree
'''
