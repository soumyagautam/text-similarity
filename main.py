from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import numpy

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    sentences = []
    sentences.append(request.form["text"])
    text2_values = request.form["text2"].split("-<!*>-")
    for value in text2_values:
        sentences.append(value)
    model_name = "sentence-transformers/all-roberta-large-v1"

    model = SentenceTransformer(model_name)
    sentence_vecs = model.encode(sentences)
    similar_percent = numpy.array(cosine_similarity(
                        [sentence_vecs[0]],
                        sentence_vecs[1:]
                      ))
    similar_percent_list = similar_percent.tolist()
    percentages = []
    for value in similar_percent_list[0]:
        percentages.append(f"{round(value, 3) * 100}%")
    indexes = []
    for percentage in percentages:
        indexes.append(percentages.index(percentage) + 1)
    return f"<p>The given text was: <br> {sentences[0]}<br><br>The similarity was as follows:<br>1. {sentences[indexes[0]]} ------ Accuracy: {percentages[0]}<br>2. {sentences[indexes[1]]} ------ Accuracy: {percentages[1]}<br>3. {sentences[indexes[2]]} ------ Accuracy: {percentages[2]}<br>4. {sentences[indexes[3]]} ------ Accuracy: {percentages[3]}<br>5. {sentences[indexes[4]]} ------ Accuracy: {percentages[4]}"

if __name__ == "__main__":  
    app.run(debug=True)
