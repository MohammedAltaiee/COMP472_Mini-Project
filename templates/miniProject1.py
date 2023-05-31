import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn import metrics
import graphviz


dataset = pd.DataFrame(
    {
        "Alt": [
            "yes",
            "yes",
            "no",
            "yes",
            "yes",
            "no",
            "no",
            "no",
            "no",
            "yes",
            "no",
            "yes",
        ],
        "Bar": [
            "no",
            "no",
            "yes",
            "no",
            "no",
            "yes",
            "yes",
            "no",
            "yes",
            "yes",
            "no",
            "yes",
        ],
        "Fri": [
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "no",
            "yes",
        ],
        "Hun": [
            "yes",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
        ],
        "Pat": [
            "some",
            "full",
            "some",
            "full",
            "full",
            "some",
            "none",
            "some",
            "full",
            "full",
            "none",
            "full",
        ],
        "Price": ["$$$", "$", "$", "$", "$$$", "$$", "$", "$$", "$", "$$$", "$", "$"],
        "Rain": [
            "no",
            "no",
            "no",
            "yes",
            "no",
            "yes",
            "yes",
            "yes",
            "yes",
            "no",
            "no",
            "no",
        ],
        "Res": [
            "yes",
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "no",
        ],
        "Type": [
            "french",
            "thai",
            "burger",
            "thai",
            "french",
            "italian",
            "burger",
            "thai",
            "burger",
            "italian",
            "thai",
            "burger",
        ],
        "Est": [
            "0-10",
            "30-60",
            "0-10",
            "10-30",
            ">60",
            "0-10",
            "0-10",
            "0-10",
            ">60",
            "10-30",
            "0-10",
            "30-60",
        ],
        "WillWait": [
            "yes",
            "no",
            "yes",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "no",
            "no",
            "yes",
        ],
    }
)


# print(dataset)
col_names = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est"]

#  converting our table to a numerical data
numericalDataset = dataset.apply(LabelEncoder().fit_transform)
# make the data and the target
data = numericalDataset[col_names]
target = numericalDataset.WillWait
# what is the test_size to include all the col_names in the results and what is the train_test_split values
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1, random_state=42)
# why the test_size is equal to 0.3?
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)  # 80 % for training
# creating the decision  classifier objects
dtc = tree.DecisionTreeClassifier(
    criterion="entropy", splitter="best"
)  # random_state=None???
# 3. train the classifier
dtc = dtc.fit(data_train, target_train)
tree.plot_tree(dtc)
dot_data = tree.export_graphviz(
    dtc,
    out_file=None,
    feature_names=[
        "Alt",
        "Bar",
        "Fri",
        "Hun",
        "Pat",
        "Price",
        "Rain",
        "Res",
        "Type",
        "Est",
    ],
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("data12")
# predict from user input
print(dtc.predict([[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]]))
