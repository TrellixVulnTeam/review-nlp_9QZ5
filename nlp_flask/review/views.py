from . import app
from flask import render_template
from flask import request

from models.review import ReviewModel

review_model = ReviewModel()

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/result', methods=('GET', 'POST'))
def form():
    if request.method == 'POST':
        line = request.form['text']
        if  len(line) > 0 and len(line) <= 1000:
            sentiment, highlight_words = review_model.predict(line, highlight=True)
            return render_template('result.html',
                                line=line,
                                highlight_words=highlight_words,
                                sentiment=sentiment)
    return render_template('welcome.html')

