from flask import Flask, render_template, request
from verifier_pipeline import HallucinationDetector

app = Flask(__name__)
detector = HallucinationDetector()


@app.route("/", methods=["GET", "POST"])
def home():
    results = None

    if request.method == "POST":
        text = request.form["text"]
        results = detector.analyze(text)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)