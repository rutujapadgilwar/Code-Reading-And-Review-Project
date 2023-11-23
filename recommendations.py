#importing python libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json

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
# Create objects for the recommended books
class Recommendations:
    def __init__(self, title, books):
        self.title = title
        self.books = books
        
class Book:
    def __init__(self, name, cover, author):
        self.name = name
        self.cover = cover
        self.author = author

# Helper method to create books in custom list from dataframe
def create_book_lists_helper(title, books):
    recommendation_books = Recommendations(title, books)
    return recommendation_books

# Recommend books by same author of the book with book_name as an input  
def recommend_books(book_name, recommendation_type):
    books_list = []
    book_name = book_name.lower()
    book_entry = df_author_recommendations[df_author_recommendations['Book-Title'].str.lower().str.contains(book_name)]

    if book_entry.empty:
        return create_book_lists_helper(f"Oops! No {recommendation_type} recommendations for the input", books_list)

    if recommendation_type == "author":
        recommendation_column = 'Book-Author'
    elif recommendation_type == "publisher":
        recommendation_column = 'Publisher'
    else:
        return create_book_lists_helper("Invalid recommendation type", books_list)

    book_value = book_entry[recommendation_column].iloc[0]
    recommendation_df = df_author_recommendations[df_author_recommendations[recommendation_column] == book_value][:5]

    recommendation_df.drop(recommendation_df.index[recommendation_df['Book-Title'] == book_name], inplace=True)

    for book_info in recommendation_df.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)

    return create_book_lists_helper(f"Top Books with same {recommendation_type}", books_list)

# Recommendation books by the same author name
def recommend_books_by_author(book_name):
    return recommend_books(book_name, "author")

# Recommendation books by the same author name
def recommend_books_by_publisher(book_name):
    return recommend_books(book_name, "publisher")

# Helper method for author name and publisher name
def recommendation_by_given_category(category_name, category_column):
    books_list = []
    category_name = category_name.lower()
    category_recommendations = df_author_recommendations.loc[pd.notna(df_author_recommendations[category_column]) & df_author_recommendations[category_column].str.lower().str.contains(category_name), :][:5]

    if category_recommendations.empty:
        return create_book_lists_helper(f"Oops! No {category_column} recommendations for the input", books_list)

    for book_info in category_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)

    return create_book_lists_helper(f"Similar top Books by given {category_column}", books_list)

# Recommendation by the given author name
def recommendation_by_given_author(author_name):
    return recommendation_by_given_category(author_name, 'Book-Author')

# Recommendation by the given publisher name
def recommendation_by_given_publisher(publisher_name):
    return recommendation_by_given_category(publisher_name, 'Publisher')

# Recommendation by similar trending similar books
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

# Get books published in the same year 
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

# Books published at the given place
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

# Books published at the given place
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

# Converting result to JSON format for frontend
def results_in_json(final_recommendations):   
    result = json.dumps(final_recommendations, default=lambda o: o.__dict__, indent=4)
    return result

# get Final results for all recommendations according to title
def get_recommendations_by_book(book_name):
    final_recommendations = [
    collaborative_recommendation(book_name) if collaborative_recommendation(book_name).books else None,
    recommend_books_by_author(book_name) if recommend_books_by_author(book_name).books else None,
    recommend_books_by_publisher(book_name) if recommend_books_by_publisher(book_name).books else None,
    recommendations_by_year(book_name) if recommendations_by_year(book_name).books else None,
    recommendation_by_same_place(book_name) if recommendation_by_same_place(book_name).books else None
    ]

    final_recommendations = [result for result in final_recommendations if result is not None]  # Remove None values

    if not final_recommendations:
        final_recommendations.append(create_book_lists_helper("No books found!", []))

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
    final_recommendations.append(recommendations_by_year(year))
    return results_in_json(final_recommendations)

def get_recommendations_by_location(location):
    final_recommendations = []
    final_recommendations.append(recommendations_by_location(location))
    return results_in_json(final_recommendations)