from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index3.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_input = ""
    inp1 = request.form['a']
    user_input = user_input + str(inp1)
    inp2 = request.form['b']
    user_input = user_input + str(inp2)
    inp3 = request.form['c']
    user_input = user_input + str(inp3)
    inp4 = request.form['d']
    user_input = user_input + str(inp4)
    inp5 = request.form['e']
    user_input = user_input + str(inp5)
    inp6 = request.form['f']
    user_input = user_input + str(inp6)
    inp7 = request.form['g']
    user_input = user_input + str(inp7)
    user_input = str(user_input)

    # 1 Preprocess
    transformed_input = transform_text(user_input)

    # 2 Vectorization

    vector_input = tfidf.transform([transformed_input])

    # 3 Predict
    result = model.predict(vector_input)[0]

    return render_template('result.html', data = result)





if __name__ == "__main__":
    app.run(debug=True)
