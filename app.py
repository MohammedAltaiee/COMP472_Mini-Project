from flask import Flask, render_template, request
from numpy import np

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        alt = request.form["alt"]
        bar = request.form["bar"]
        fri = request.form["fri"]
        hun = request.form["hun"]
        pat = request.form["pat"]
        price = request.form["price"]
        rain = request.form["rain"]
        res = request.form["res"]
        typ = request.form["typ"]
        est = request.form["est"]

    print(alt, bar)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
