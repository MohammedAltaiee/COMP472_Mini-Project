from flask import Flask, render_template, request, session, url_for, redirect
import pandas as pd
import graphviz
from graphviz import render
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import math


app = Flask(__name__)
app.secret_key= "thekeyisthekey"

@app.route("/", methods=["POST", "GET"])
def user_input():

    # Getting user input from form (UI)
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

    # Saving dataset
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

        # Saving column names from data set 
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
        
        # Converting dataset table to a numerical data
        numericalDataset = dataset.apply(LabelEncoder().fit_transform)
        print(numericalDataset)

        # Assigning columns names to data and target to last column of the dataset 
        data = numericalDataset[col_names]
        target = numericalDataset.WillWait
        
        # Split arrays or matrices into random train and test subsets
        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=0.2, random_state=32)  # 80 % for training, 20 % for testing
        
        # Creating the decision tree classifier
        dtc = tree.DecisionTreeClassifier(
            criterion="entropy", splitter="best")
         
        predDataset = [alt, bar, fri, hun, price, pat, rain, res, type, est]
        print(predDataset)

        # Training the decision tree classifier. Training using fit() method by passing input data (data_train) as a X and output (data target_train) as Y
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

        # Printing the tree
        graph = graphviz.Source(dot_data, format="png")
        
        # Saving the png in static/data directory
        graph.render("static/data")

        # Predicting from user input
        if dtc.predict([predDataset]):
            # className="yes"
            session['classN']= 'Yes'
            # return 'classN'
        else:
            # className="no"
            session['classN']= 'No'
            # return 'classN'



        # Calculating the entropy of the last column "Will wait"
        print(dataset["WillWait"])
        counterYes=0
        counterNo=0
        
        for item in dataset["WillWait"]:
            if item == "yes":
                counterYes+=1
            else:
                counterNo+=1

        print("counterYes: "+str(counterYes))
        print("counterNo: "+str(counterNo))
        # Calculating the entropy using the formula given in class
        entropy=-(counterNo/(counterNo+counterYes))*math.log2(counterNo/(counterNo+counterYes))-(counterYes/(counterNo+counterYes))*math.log2(counterYes/(counterNo+counterYes))

        print(f"The entropy of the last column is equal to {entropy}")
         
        return redirect(url_for("output"))

    return render_template("user_input.html")

@app.route("/output")
def output():
    # Passing the value of the class name from the user page 
    cn=session.get('classN',None)
    print("Output decision :"+cn)
    return render_template("output.html", cn=cn)


if __name__ == "__main__":
    app.run(debug=True)
