# Imports
from flask import Flask, request, jsonify
import nltk
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import csv
import numpy as np
from flask_restful import Resource, Api
from flask_cors import CORS


# Downloads needed for lemmatization
# nltk.download('wordnet')
# nltk.download('omw-1.4')



app = Flask(__name__)
api = Api(app)
CORS(app)


# Load models
tfidf = joblib.load("./models/tfidf.pkl")
d2vModel = Doc2Vec.load("./models/d2v.model", allow_pickle=True)

# Parts of speech tagging
def posCount(text, wordCount):
    
    # Tokenize the words in the text
    tokens = nltk.word_tokenize(text)
    
    # Assign POS tags to each words
    pos = nltk.pos_tag(tokens, tagset='universal')
    
    # Count the POS tags
    counts = Counter(tag for _, tag in pos)
    
    # Get individual counts for POS of interests
    noun = counts["NOUN"] / wordCount
    verb = counts["VERB"] / wordCount
    adjective = counts["ADJ"] / wordCount
    adverb = counts["ADV"] / wordCount
    
    return noun, verb, adjective, adverb

# handle TFIDF results
def processTFIDF(data, X_tf):
    # Read the BoW file
    with open('./models/BoW.csv', newline='') as f:
        reader = csv.reader(f)
        BoW = (np.array(list(reader))).flatten()
    wc = pd.DataFrame.sparse.from_spmatrix(X_tf, columns=tfidf.get_feature_names_out())
    wc_dict = wc.to_dict()
    for key in wc_dict.keys():
        if (key in BoW):
            data["BOW"][f'{key}'] = wc_dict[f'{key}'][0]

    return data

# Get Doc2Vec Inference
def getDoc2Vec(text):
    vector = d2vModel.infer_vector(text.split(" "))
    return vector.tolist()

class aiModel(Resource):
    def post(self):
        wn = nltk.WordNetLemmatizer()
        # POS tagging
        data = request.json
        data_type = (request.args).get("type")
        data["clean_text"] = ' '.join([wn.lemmatize(word) for word in data["clean_text"].split(' ')])
        data["unique_clean_text"] = " ".join(dict.fromkeys(data["clean_text"].split()))
        data["word_count"] = len(data["unique_clean_text"].split())

        pos_counts = posCount(data["unique_clean_text"], data["word_count"])

        data["norm_noun"] = pos_counts[0]
        data["norm_verb"] = pos_counts[1]
        data["norm_adj"] = pos_counts[2]
        data["norm_adv"] = pos_counts[3]
        data["BOW"] = {}

        # sentiment analysis
        va = SentimentIntensityAnalyzer()
        data["compound_sent"] =  va.polarity_scores(data["clean_text"])['compound']

        # Run TFIDF
        X_tf = tfidf.transform([data["clean_text"]])
        data = processTFIDF(data, X_tf)

        # Process Doc2Vec
        data["post2vec"] = getDoc2Vec(data["clean_text"])
        # Pass to model
        # TODO Get results from model
        results = {
            "poor_mental_health": True
        }
        return (jsonify(results))
    
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

api.add_resource(aiModel, '/')

if __name__ == '__main__':
    app.run()