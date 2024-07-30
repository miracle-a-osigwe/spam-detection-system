import collections.abc
#pattern needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import joblib
import re
import requests
from bs4 import BeautifulSoup as scraper
from pattern.web import find_urls
from flask import Flask, request, render_template
from flask_restful import reqparse, abort, Api, Resource
import sys
import os

try:
    import nltk
    nltk.data.path.insert(0, os.path.abspath("nltk data/"))
    
except ConnectionError:
    print('You need internet connection to run this for the first time.')
    print('Connect to internet and try again')
    sys.exit()
else:
    from nltk.corpus import stopwords
finally:
    from nltk.stem.porter import PorterStemmer

app = Flask(__name__)


stem = PorterStemmer() #text stemming
stopwordSet = set(stopwords.words('english')) #create a set of the stopwords

replace={'title': '', 'h1': '', 'h2': '', 'h3': '', 'h4': '', 'h5': '', 
         'h6': '', 'p': '', 'center': '', 'blockquote': '', 'div': '', 
         'table': '', 'ul': '', 'ol': '', 'dl': '', 'pre': '', 'code': '', 
         'form': '', 'br': '', 'tr': '', 'td': '', 'li': '', 'img': '', 'th': '',
        '>': '', '<': '', '/': ''}

def preprocess(text):
    data = text
    for i in range(len(data)):
        # Extract links from the text
        links = find_urls(data[i], unique=True)
        for link in links:
            if link[:4].lower() != 'http': # Check if the link contacts http
                link = "http://" + link
            try:
                response = requests.get(link)
                soup = scraper(response.content, 'html5lib')
                textBody = soup.find('body')
                body = re.findall(r"[\w]+", textBody.text)
                text = ' '.join(body)
                for x in list(replace.keys()):
                    text = re.sub(x, "", text)
            except Exception as e:
                response = e.__str__()
                text = "Invalid website"
            
            data[i].replace(link, text)
            data[i] = data[i] + link
        
        message = data[i].lower() #using reqular expression to clean the text
        message = message.split() #split the words
        message = [stem.stem(word) for word in message if not word in stopwordSet]# stem each word not found in the stopword set
        data[i] = ' '.join(message) #join the data
    return data #return the data

# Prediction function
def predictions(text):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    vectorized_text = vectorizer.transform(text)
    predict = model.predict(vectorized_text)
    return predict

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        text = [x for x in request.form.values()]
        links = find_urls(text)
        default_text = text.copy()
        text_result = preprocess(text)
        if (len(text_result[0]) == 0):
            return render_template('index.html', prediction_text="No message entered", review_text="")
        else:
            result = predictions(text_result)
            output = 'Not a Spam' if result[0] == 0 else 'a Spam'
            resp = "text" if len(links) == 0 else "text contain link(s) and"
            return render_template('index.html', prediction_text=f"The {resp} is {output}", review_text=f"{default_text[0]}\n{text_result}")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)