from flask import Flask,render_template, request
import pickle
import numpy as np
import json

from recommendations import getAllRecommendationsByBookName
from recommendations import getAllRecommendationsByAuthorName
from recommendations import getAllRecommendationsByPublisherName
from recommendations import getAllRecommendationsByYear
from recommendations import getAllRecommendationsByLocation
from recommendations import loadData
top_books = pickle.load(open('top_books.pkl', 'rb'))
loadData()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html',
    book_name = list(top_books['Book-Title'].values),
    book_author = list(top_books['Book-Author'].values),
    book_image = list(top_books['Image-URL-M'].values)
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('searchBooks.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    userInput = request.form.get('userInput')
    selectorValue = request.form.get('searchBy')
    if len(str(userInput)) == 0:   
        return render_template('searchBooks.html')
    allResults = []

    match selectorValue:
        case "bookname":
            allResults = json.loads(getAllRecommendationsByBookName(userInput))
        
        case "author":
            allResults = json.loads(getAllRecommendationsByAuthorName(userInput))

        case "publisher":
            allResults = json.loads(getAllRecommendationsByPublisherName(userInput))

        case "year":
            allResults = json.loads(getAllRecommendationsByYear(userInput))

        case "location":
            allResults = json.loads(getAllRecommendationsByLocation(userInput))

    return render_template('searchBooks.html', bookList=allResults)


if __name__== '__main__':
    app.run(debug=True)