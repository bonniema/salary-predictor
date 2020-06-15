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

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

#Load the vectorizer and model here

labels=['Below $50,000','$50,000-$70,000','$70,000-$90,000','$90,000-$120,000','$120,000-$150,000','$150,000 and Above']
model = load('finalized_model_2.joblib')
tfidf_vectorizer = pickle.load(open('fitted_vectorizer.pickle','rb')) 

###
# Routing for your application.
###

@app.route('/', methods=['GET'])
def home():
    """Render website's home page."""
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def salary_predictor():
	description = request.form.get('description')
	result = model.predict(tfidf_vectorizer.transform([description]))

	stopwords = set(STOPWORDS)
	stopwords.update(["to","sex","may","Ability to","Full time","Experience with", "Job Type"])

	wordcloud = WordCloud(stopwords = stopwords,max_font_size=50, max_words=100, background_color='white').generate(description)

	
	plt.figure(figsize=(8,6))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')

	#Save it to a temporary buffer
	buf = io.BytesIO()
	plt.savefig(buf, format="png")
	buf.seek(0)

	#Embed the result in the html output
	img = base64.encodebytes(buf.getvalue()).decode("ascii")
	return jsonify(salaryRange=labels[result[0]], word_cloud_image=img)
	




"""return render_template('home.html', salaryRange = labels[result[0]], salary_prediction_text="The salary range of this job:")
"""

if __name__ == '__main__':
    app.run(debug=True)
