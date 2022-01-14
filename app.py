#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:43:52 2022

@author: abhinav
"""

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import json
import pickle
import nltk 
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__,  template_folder='template')

# Unpickling the machine learning model and storing it into a variable 
model = pickle.load(open("model.pkl","rb"))

#Creating an object for lemmtization
lemmatizer = WordNetLemmatizer()

@app.route("/",methods=['GET'])
def home():
    return render_template("/index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method == "POST":
        url = request.form['url']
        headline = request.form['headline']
        body = request.form['body']
        
        input_list = [url,headline,body]
        empty_input_list = []
       
        def text_preprocessing(empty_input_list,input_list):
        
        # Merging all the feature into one sentence and appending it into the list
            empty_input_list.append(" ".join(str(sentence) for sentence in input_list))

            # Converting all the letters in the sentence to lower caps
            empty_input_list = [x.lower() for x in empty_input_list]
            
            #Splitting the sentences into words by using word tokenize and removing the stopwords simaltaneously applying
            #lemmatizer to words
            word_tokens = nltk.word_tokenize(empty_input_list[0])
            lemmatized_tokens = [lemmatizer.lemmatize(x) for x in word_tokens if x not in set(stopwords.words("english"))]
            empty_input_list.clear()
            empty_input_list = " ".join(lemmatized_tokens)
            
            #Elminating punctuations from sentences as they do not add value. 
            for x in empty_input_list:
                if x in string.punctuation:
                    empty_input_list = empty_input_list.replace(x," ")
                    
        text_preprocessing(empty_input_list,input_list)
        
        vectorizer = pickle.load(open("vectorizer.pkl","rb"))
        
        model_input = vectorizer.transform(empty_input_list)
        
        prediction = model.predict(model_input)
            
        
        
        if prediction == 1:
            return render_template("/index.html",prediction_text = "The news article is genuine")
        else:
            return render_template("/index.html",prediction_text = "The news article is fake")
        
if __name__ == "__main__" :
    app.run(debug=True)








