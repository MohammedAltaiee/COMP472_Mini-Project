import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import preprocessing
import pydotplus

dataset = pd.DataFrame({

    'Alt':['yes', 'yes','no','yes','yes','no','no','no','no','yes','no','yes'],
    'Bar':['no','no','yes','no','no','yes','yes','no','yes','yes','no','yes'],
    'Fri':['no','no','no','yes','yes','no','no','no','yes','yes','no','yes'],
    'Hun':['yes','yes','no','yes','no','yes','no','yes','no','yes','no','yes'],
    'Pat':['some','full','some','full','full','some','none','some','full','full','none','full'],
    'Price':['$$$','$','$','$','$$$','$$','$','$$','$','$$$','$','$'],
    'Rain':['no','no','no','yes','no','yes','yes','yes','yes','no','no','no'],
    'Res':['yes','no','no','no','yes','yes','no','yes','no','yes','no','no'],
    'Type': ['french', 'thai', 'burger', 'thai', 'french', 'italian', 'burger', 'thai', 'burger', 'italian', 'thai', 'burger'],
    'Est':['0-10','30-60','0-10','10-30','>60','0-10','0-10','0-10','>60','10-30','0-10','30-60'],
    'WillWait': ['yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes'
           ]
    })
numericalDataset = dataset.apply(LabelEncoder().fit_transform)
print(numericalDataset)
# how to train the classifier to build the tree
# 1. split the dataset into features and target variable
features = numericalDataset.iloc[:,0:10]
target = numericalDataset.iloc[:,0:11]
# 2. create the decision  classifier object
clf = tree.DecisionTreeClassifier(criterion="entropy")

# 3. train the classifier
clf = clf.fit(features,target)
# 4. predict the target variable for the test data
print(clf.predict([[1,0,0,1,1,0,0,1,0,0]]))
# get the string value for the clf
print(clf.classes_)
# get the probability for the clf
print(clf.predict_proba([[1,0,0,1,1,0,0,1,0,0]]))
# get the feature importance
print(clf.feature_importances_)
# 5. visualize the classifier
tree.plot_tree(clf)
# 6. save the tree
tree.export_graphviz(clf, out_file='tree.dot', feature_names=['Alt','Bar','Fri','Hun','Pat','Price','Rain','Res','Type','Est'], class_names=['WillWait'], filled=True, rounded=True, special_characters=True)
# 5. visualize the tree
tree.plot_tree(clf)
# 6. save the tree
tree.export_graphviz(clf, out_file='tree.dot', feature_names=['Alt','Bar','Fri','Hun','Pat','Price','Rain','Res','Type','Est'], class_names=['WillWait'], filled=True, rounded=True, special_characters=True)
# 7. convert the tree to png
from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['Alt','Bar','Fri','Hun','Pat','Price','Rain','Res','Type','Est'], class_names=['WillWait'], filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')
Image(graph.create_png())
# 8. predict the target variable for the test data
print(clf.predict([[0,0,0,0,1,0,0,1,0,0]]))
# 8. open the png file
# 9. convert the tree to pdf
graph.write_pdf('tree.pdf')
# 10. open the pdf file
# 11. convert the tree to svg

