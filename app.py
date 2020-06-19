"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, request, redirect, Response, url_for

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import PIL


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
from matplotlib.figure import Figure
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from joblib import dump, load
from flask.json import jsonify

app = Flask(__name__, static_url_path="/static")

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

#Load the vectorizer and model here

labels=['Below $50,000','$50,000-$70,000','$70,000-$90,000','$90,000-$120,000','$120,000-$150,000','$150,000 and Above']
model = load('finalized_model_2.joblib')
tfidf_vectorizer = pickle.load(open('fitted_vectorizer.pickle','rb')) 

###
# Routing for your application.
###

@app.route('/')
def home():
    """Render website's home page."""
    return render_template('home.html', salaryRange="", salary_prediction_text="", show_word_cloud=False)


@app.route('/predict', methods=['POST'])
def salary_predictor():
	description = request.form.get('description')
	result = model.predict(tfidf_vectorizer.transform([description]))
	
	return render_template('home.html', salaryRange = labels[result[0]], show_word_cloud=True, salary_prediction_text="The salary range of this job:")


@app.route('/predict')
def generate_word_cloud(description):

	
	stopwords = set(STOPWORDS)
	stopwords.update(["to","sex","may","Ability to","Full time","Experience with", "Job Type"])

	wordcloud = WordCloud(stopwords = stopwords, max_font_size=50, max_words=100, background_color='white').generate(description)

	img = io.StringIO()
	plt.figure(figsize=(8,6))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	#Save it to a temporary buffer
	plt.savefig(img, format='png')
	img.seek(0)

	plot_url = base64.b64encode(img.getvalue())
	return render_template('home.html', plot_url=plot_url)
	


"""

@app.route('/predict', methods=['POST'])
def salary_predictor():
	description = request.form.get('description')
	result = model.predict(tfidf_vectorizer.transform([description]))
	generate_word_cloud(description)
	return render_template('home.html', salaryRange = labels[result[0]], url="/static/images/word_cloud.png", show_word_cloud=True, salary_prediction_text="The salary range of this job:")



def generate_word_cloud(description):

	
	stopwords = set(STOPWORDS)
	stopwords.update(["to","sex","may","Ability to","Full time","Experience with", "Job Type"])

	wordcloud = WordCloud(stopwords = stopwords, max_font_size=50, max_words=100, background_color='white').generate(description)

	plt.figure(figsize=(8,6))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	#Save it to a temporary buffer
	plt.savefig('static/images/word_cloud.png')
	


return render_template('home.html', salaryRange = labels[result[0]], salary_prediction_text="The salary range of this job:")
img = base64.encodebytes(buf.getvalue()).decode("ascii")
"""

if __name__ == '__main__':
    app.run(debug=True)
