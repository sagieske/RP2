# DOES NOT WORK YET
# CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.

from sklearn import tree
from sklearn.externals.six import StringIO
import os

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# output to graphviz tree
with open('iris.dot', 'w') as f:
	f = tree.export_graphviz(clf, out_file=f)

# output to PDF (ot -Tpdf iris.dot -o iris.pdf)
os.unlink('iris.dot')
