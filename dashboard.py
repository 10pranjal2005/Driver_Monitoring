from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

LOG_FILE = "fatigue_log.csv"


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/data")
def get_data():

    if os.path.exists(LOG_FILE):

        df = pd.read_csv(LOG_FILE)

        latest = df.tail(1).to_dict(orient="records")[0]

        history = df.tail(50).to_dict(orient="records")

        return jsonify({
            "latest": latest,
            "history": history
        })

    return jsonify({"error": "No data yet"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)