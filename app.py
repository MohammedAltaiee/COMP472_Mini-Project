from flask import Flask, render_template, request, session, url_for, redirect
import pandas as pd
import graphviz
from graphviz import render
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# from numpy import np

app = Flask(__name__)
app.secret_key= "thekeyisthekey"

@app.route("/", methods=["POST", "GET"])
def user_input():
    if request.method == "POST":
        alt = int(request.form.get("alt"))
        bar = int(request.form.get("bar"))
        fri = int(request.form.get("fri"))
        hun = int(request.form.get("hun"))
        pat = int(request.form.get("pat"))
        price = int(request.form.get("price"))
        rain = int(request.form.get("rain"))
        res = int(request.form.get("res"))
        type = int(request.form.get("type"))
        est = int(request.form.get("est"))

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
                "Price": [
                    "$$$",
                    "$",
                    "$",
                    "$",
                    "$$$",
                    "$$",
                    "$",
                    "$$",
                    "$",
                    "$$$",
                    "$",
                    "$",
                ],
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
        col_names = [
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
        ]
        
        #  converting our table to a numerical data
        # numericalperdDataset = predDataset.apply(LabelEncoder().fit_transform)
        numericalDataset = dataset.apply(LabelEncoder().fit_transform)
        print(numericalDataset)
        # make the data and the target
        data = numericalDataset[col_names]
        target = numericalDataset.WillWait
        # what is the test_size to include all the col_names in the results and what is the train_test_split values
        # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1, random_state=42)
        # why the test_size is equal to 0.3?
        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=0.2, random_state=32
        )  # 80 % for training
        # creating the decision  classifier objects
        dtc = tree.DecisionTreeClassifier(
            criterion="entropy", splitter="best"
        )  # random_state=None???
        predDataset = [alt, bar, fri, hun, price, pat, rain, res, type, est]
        print(predDataset)
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
            class_names=["No", "Yes"],
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data, format="png")
        graph.render("static/data")

        # # predict from user input
        if dtc.predict([predDataset]):
            # className="yes"
            session['classN']= 'yes'
            # return 'classN'
        else:
            # className="no"
            session['classN']= 'No'
            # return 'classN'
        
        # return className
        # print(session["classN"])
        #session['className'] = className
        return redirect(url_for("output"))

    return render_template("user_input.html")

@app.route("/output")
def output():
    cn=session.get('classN',None)
    print(cn)
    # className = session.get('className')
    print("output def :"+cn)
    return render_template("output.html", cn=cn)


if __name__ == "__main__":
    app.run(debug=True)
