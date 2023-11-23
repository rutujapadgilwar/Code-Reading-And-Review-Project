# ## Data Loading

#importing python libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# %%
df_recommendation_dataset = pd.DataFrame()
df_author_recommendations = pd.DataFrame()
df_books = pd.DataFrame()
df_pivot_table = pd.DataFrame()
df_similarity_scores = []
def load_data():
    global df_recommendation_dataset, df_author_recommendations, df_books, df_pivot_table, df_similarity_scores   
    df_recommendation_dataset = (pickle.load(open('pklFiles/books_with_ratings.pkl', 'rb')))
    df_author_recommendations = (pickle.load(open('pklFiles/author_recommendations.pkl', 'rb')))
    df_books = (pickle.load(open('pklFiles/books.pkl', 'rb')))
    df_pivot_table = (pickle.load(open('pklFiles/pivot_table.pkl', 'rb')))
    df_similarity_scores = pickle.load(open('pklFiles/similarity_scores.pkl', 'rb'))

load_data()
# #### Create objects for the recommended books

class Recommendations:
    def __init__(self, title, books):
        self.title = title
        self.books = books
        
class Book:
    def __init__(self, name, cover, author):
        self.name = name
        self.cover = cover
        self.author = author

# #### Helper method to create books in custom list from dataframe

import json
def create_book_lists_helper(title, books):
    recommendation_books = Recommendations(title, books)
    return recommendation_books

# ## Recommendation for same author

#Recommend books by same author of the book with book_name as an input  
def recommendation_by_same_author(book_name):
    books_list = []
    print(book_name)
    book_name = book_name.lower()
    book_entry = df_author_recommendations[df_author_recommendations['Book-Title'].str.lower().str.contains(book_name)]
    if book_entry.empty:
        return create_book_lists_helper("oops! No author recommendations for the input", books_list)
    book_author = book_entry['Book-Author'].iloc[0]
    author_recommendations = df_author_recommendations.loc[df_author_recommendations['Book-Author'] == book_author,:][:5]
    author_recommendations.drop(author_recommendations.index[author_recommendations['Book-Title'] == book_name], inplace = True)
    for book_info in author_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    
    return create_book_lists_helper("Top Books with same author", books_list)

# ## Recommendation by the given author name
def recommendation_by_given_author(author_name):
    books_list = []
    author_name = author_name.lower()
    author_recommendations = df_author_recommendations.loc[pd.notna(df_author_recommendations['Book-Author']) & df_author_recommendations['Book-Author'].str.lower().str.contains(author_name), :][:5]
    if author_recommendations.empty:
        return create_book_lists_helper("oops! No author recommendations for the input", books_list)
    for book_info in author_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    return create_book_lists_helper("Similar top Books by given author", books_list)

# ## Recommendation for same publisher
#Recommend books by same publisher of the book with bookname as an input  
def recommendation_by_same_publisher(book_name):
    books_list = []
    book_name = book_name.lower()
    book_entry = df_author_recommendations[df_author_recommendations['Book-Title'].str.lower().str.contains(book_name)]
    if book_entry.empty:
        return create_book_lists_helper("oops! No publisher recommendations for the input", books_list)
    book_publisher = book_entry['Publisher'].iloc[0]
    publisher_recommendations = df_author_recommendations.loc[df_author_recommendations['Publisher'] == book_publisher,:][:5]
    publisher_recommendations.drop(publisher_recommendations.index[publisher_recommendations['Book-Title'] == book_name], inplace = True)

    for book_info in publisher_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    return create_book_lists_helper("Top Books published by same publisher", books_list) 

# ## Recommendation by the given publisher name
#recommendation by the given publisher name
def recommendation_by_given_publisher(publisher_name):
    books_list = []
    publisher_name = publisher_name.lower()
    author_recommendations = df_author_recommendations.loc[df_author_recommendations['Publisher'].str.lower().str.contains(publisher_name),:][:5]
    if author_recommendations.empty:
        return create_book_lists_helper("oops! No publisher recommendations for the input", books_list)
    for book_info in author_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    return create_book_lists_helper("Similar top Books by given publisher", books_list)

#similarity-scores
from sklearn.metrics.pairwise import cosine_similarity
#df_similarity_scores = cosine_similarity(df_pivot_table)

# %%
def collaborative_recommendation(book_name):
    books_list = []
    array_size = np.where(df_pivot_table.index== book_name)[0]
    if array_size.size == 0:
        return create_book_lists_helper("oops! No trending recommendations for the input", books_list)
    book_index = np.where(df_pivot_table.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(df_similarity_scores[book_index])),key=lambda x:x[1],reverse=True)[1:6]
    if len(similar_books) == 0:
        return create_book_lists_helper("Top trending similar books", books_list)
   
    for book_info in similar_books:
        temp_df = df_books[df_books['Book-Title'] == df_pivot_table.index[book_info[0]]]
        book_title = temp_df.drop_duplicates('Book-Title')['Book-Title'].values
        book_author = temp_df.drop_duplicates('Book-Title')['Book-Author'].values
        cover_image = temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values
        recommended_book = Book(book_title[0], cover_image[0], book_author[0])
        books_list.append(recommended_book)
    
    return create_book_lists_helper("Top trending similar books", books_list)


# ## Books Published Yearly
# get books published in the same year 
def recommendations_by_year(year_or_book: int or str):
    books_list = []
    try:
        year_of_publication = int(year_or_book)
        #valid year checking
        if (year_of_publication < 1900):
            return create_book_lists_helper("oops! Please input the valid year between 1900 - 2022", books_list)
        elif (year_of_publication > 2022):
            return create_book_lists_helper("oops! Please input the valid year between 1900 - 2022", books_list)
        
        #filter books in the same year
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Year-Of-Publication'] == year_of_publication]
    except:
        #check for book name
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Book-Title'].str.lower().str.contains(year_or_book.lower())]
        
        #no books from the same year
        if (len(same_year_books)== 0):
            return create_book_lists_helper("oops! No yearly recommendations for the input", books_list)        
        
        #year of publication of the same book
        year_of_publication = same_year_books.iloc[0]['Year-Of-Publication']
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Year-Of-Publication'] == year_of_publication]

    if (len(same_year_books)== 0):
        return create_book_lists_helper("oops! No recommendations for year input", books_list)

    #top 5 rated books
    same_year_books = same_year_books.sort_values(by="Book-Rating", ascending=False)[:5]
    
    #dropping the duplicates
    same_year_books = same_year_books.drop_duplicates(subset=["Book-Title"])
    for book_info in same_year_books.values.tolist():
        recommended_book = Book(book_info[0], book_info[4], book_info[1])
        books_list.append(recommended_book)
    return create_book_lists_helper("Trending books in the same year", books_list)

# ## Books published at the given place
#location as input
def recommendations_by_location(place):
    books_list = []
    if place is not None:
        place = place.lower()
    
    places = ((df_recommendation_dataset['City'].str.lower() == place) |
            (df_recommendation_dataset['State'].str.lower() == place) |
            (df_recommendation_dataset['Country'].str.lower() == place))
    
    if places.any():
        same_place_books = df_recommendation_dataset[places]
        #top 5 rated books
        same_place_books = same_place_books.sort_values(by = "Book-Rating", ascending=False)[:5]
        same_place_books = same_place_books.drop_duplicates(subset=["Book-Title"])
        if(len(same_place_books) == 0):
            return create_book_lists_helper("oops! No recommendations for place input", books_list)
        for book_info in same_place_books.values.tolist():
            recommended_book = Book(book_info[0], book_info[4], book_info[1])
            books_list.append(recommended_book)
        return create_book_lists_helper("Trending books at the same location", books_list)
    else:
        return create_book_lists_helper("oops! No recommendations for place input", books_list)

# %%
#book name as input
def recommendation_by_same_place(book_name):
    books_list = []
    if book_name is not None:
        book_name = book_name.lower()

        #check for book name
        same_place_books = df_recommendation_dataset[df_recommendation_dataset['Book-Title'].str.lower().str.contains(book_name.lower())]
        
        #no books from the same year
        if (len(same_place_books) == 0):
            return create_book_lists_helper("oops! No recommendations for place input", books_list)
            
    places = ((df_recommendation_dataset['City'].str.lower() == same_place_books.iloc[0]['City'].lower()) |
                  (df_recommendation_dataset['State'].str.lower() == same_place_books.iloc[0]['State'].lower()) |
                  (df_recommendation_dataset['Country'].str.lower() == same_place_books.iloc[0]['Country'].lower()))
    
    if places.any():
        same_place_books = df_recommendation_dataset[places]
        #top 5 rated books
        same_place_books = same_place_books.sort_values(by = "Book-Rating", ascending=False)[:5]
        same_place_books = same_place_books.drop_duplicates(subset=["Book-Title"])
        if(len(same_place_books) == 0):
            return create_book_lists_helper("oops! No recommendations for place input", books_list)    
        for book_info in same_place_books.values.tolist():
            recommended_book = Book(book_info[0], book_info[4], book_info[1])
            books_list.append(recommended_book)
        return create_book_lists_helper("Trending books at the same location", books_list)
    else:
        return create_book_lists_helper("oops! No recommendations for place input", books_list)    

# ### Converting result to JSON format for frontend
def results_in_json(final_recommendations):   
    result = json.dumps(final_recommendations, default=lambda o: o.__dict__, indent=4)
    return result

# get Final results for all recommendations according to title
def get_recommendations_by_book(book_name):
    final_recommendations = []
    results = collaborative_recommendation(book_name)
    if len(results.books) > 0:
        final_recommendations.append(collaborative_recommendation(book_name))
    results = recommendation_by_same_author(book_name)
    if len(results.books) > 0:
        final_recommendations.append(recommendation_by_same_author(book_name))
    results = recommendation_by_same_publisher(book_name)
    if len(results.books) > 0:
        final_recommendations.append(recommendation_by_same_publisher(book_name))
    results = recommendations_by_year(book_name)
    if len(results.books) > 0:
        final_recommendations.append(recommendations_by_year(book_name))
    results = recommendation_by_same_place(book_name)
    if len(results.books) > 0:
        final_recommendations.append(recommendation_by_same_place(book_name))

    if len(final_recommendations) == 0:
        final_recommendations.append(create_book_lists_helper("No books found!",[]))
    
    return results_in_json(final_recommendations)

def get_recommendations_by_author(author_name):
    final_recommendations = []
    final_recommendations.append(recommendation_by_given_author(author_name))
    return results_in_json(final_recommendations)

def get_recommendations_by_publisher(publisher_name):
    final_recommendations = []
    final_recommendations.append(recommendation_by_given_publisher(publisher_name))
    return results_in_json(final_recommendations)

def get_recommendations_by_year(year):
    final_recommendations = []
    final_recommendations.append(get_recommendations_by_year(year))
    return results_in_json(final_recommendations)

def get_recommendations_by_location(location):
    final_recommendations = []
    final_recommendations.append(recommendations_by_location(location))
    return results_in_json(final_recommendations)
