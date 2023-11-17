from flask import Flask,render_template, request
import pickle
import numpy as np
import json

from recommendations import get_recommendations_by_book
from recommendations import get_recommendations_by_author
from recommendations import get_recommendations_by_publisher
from recommendations import get_recommendations_by_yearr
from recommendations import get_recommendations_by_location
from recommendations import load_data

top_50_books = pickle.load(open('pklFiles/top_50_books.pkl', 'rb'))
load_data()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html',
    book_name = list(top_50_books['Book-Title'].values),
    book_author = list(top_50_books['Book-Author'].values),
    book_image = list(top_50_books['Image-URL-M'].values)
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('searchBooks.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    option_selection = request.form.get('searchBy')
    if len(str(user_input)) == 0:   
        return render_template('searchBooks.html')
    final_results = []

    match option_selection:
        case "bookname":
            final_results = json.loads(get_recommendations_by_book(user_input))
        
        case "author":
            final_results = json.loads(get_recommendations_by_author(user_input))

        case "publisher":
            final_results = json.loads(get_recommendations_by_publisher(user_input))

        case "year":
            final_results = json.loads(get_recommendations_by_yearr(user_input))

        case "location":
            final_results = json.loads(get_recommendations_by_location(user_input))

    return render_template('searchBooks.html', bookList=final_results)


if __name__== '__main__':
    app.run(debug=True)