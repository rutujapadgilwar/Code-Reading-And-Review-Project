# %% [markdown]
# ## Data Loading

# %%
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
def loadData():
    global df_recommendation_dataset, df_author_recommendations, df_books, df_pivot_table, df_similarity_scores   
    df_recommendation_dataset = (pickle.load(open('books_with_ratings.pkl', 'rb')))
    df_author_recommendations = (pickle.load(open('author_recommendations.pkl', 'rb')))
    df_books = (pickle.load(open('books.pkl', 'rb')))
    df_pivot_table = (pickle.load(open('pivot_table.pkl', 'rb')))
    df_similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
loadData()
# %% [markdown]
# #### Create objects for the recommended books

# %%
class Recommendations:
    def __init__(self, title, books):
        self.title = title
        self.books = books
        
class Book:
    def __init__(self, name, cover, author):
        self.name = name
        self.cover = cover
        self.author = author

# %% [markdown]
# #### Helper method to create books in custom list from dataframe

# %%
import json
def create_book_lists_helper(title, books):
    recommendation_books = Recommendations(title, books)
    return recommendation_books

# %% [markdown]
# ## Recommendation for same author

# %%
#Recommend books by same author of the book with bookname as an input  
def recommendation_by_same_author(bookname):
    booksList = []
    bookname = bookname.lower()
    book_entry = df_author_recommendations[df_author_recommendations['Book-Title'].str.lower().str.contains(bookname)]
    if book_entry.empty:
        return create_book_lists_helper("oops! No author recommendations for the input", booksList)
    book_author = book_entry['Book-Author'].iloc[0]
    author_recommendations = df_author_recommendations.loc[df_author_recommendations['Book-Author'] == book_author,:][:5]
    author_recommendations.drop(author_recommendations.index[author_recommendations['Book-Title'] == bookname], inplace = True)
    for book in author_recommendations.values.tolist():
        rBook = Book(book[0], book[8], book[5])
        booksList.append(rBook)
    
    return create_book_lists_helper("Top Books with same author", booksList)
    

# %% [markdown]
# ## Recommendation by the given author name

# %%
def recommendation_by_given_author(authorName):
    booksList = []
    authorName = authorName.lower()
    author_recommendations = df_author_recommendations.loc[pd.notna(df_author_recommendations['Book-Author']) & df_author_recommendations['Book-Author'].str.lower().str.contains(authorName), :][:5]
    if author_recommendations.empty:
        return create_book_lists_helper("oops! No author recommendations for the input", booksList)
    for book in author_recommendations.values.tolist():
        rBook = Book(book[0], book[8], book[5])
        booksList.append(rBook)
    return create_book_lists_helper("Similar top Books by given author", booksList)

# %% [markdown]
# ## Recommendation for same publisher

# %%
#Recommend books by same publisher of the book with bookname as an input  
def recommendation_by_same_publisher(bookname):
    booksList = []
    bookname = bookname.lower()
    book_entry = df_author_recommendations[df_author_recommendations['Book-Title'].str.lower().str.contains(bookname)]
    if book_entry.empty:
        return create_book_lists_helper("oops! No publisher recommendations for the input", booksList)
    book_publisher = book_entry['Publisher'].iloc[0]
    publisher_recommendations = df_author_recommendations.loc[df_author_recommendations['Publisher'] == book_publisher,:][:5]
    publisher_recommendations.drop(publisher_recommendations.index[publisher_recommendations['Book-Title'] == bookname], inplace = True)

    for book in publisher_recommendations.values.tolist():
        rBook = Book(book[0], book[8], book[5])
        booksList.append(rBook)
    return create_book_lists_helper("Top Books published by same publisher", booksList) 

# %% [markdown]
# ## Recommendation by the given publisher name

# %%
#recommendation by the given publisher name
def recommendation_by_given_publisher(publisherName):
    booksList = []
    publisherName = publisherName.lower()
    author_recommendations = df_author_recommendations.loc[df_author_recommendations['Publisher'].str.lower().str.contains(publisherName),:][:5]
    if author_recommendations.empty:
        return create_book_lists_helper("oops! No publisher recommendations for the input", booksList)
    for book in author_recommendations.values.tolist():
        rBook = Book(book[0], book[8], book[5])
        booksList.append(rBook)
    return create_book_lists_helper("Similar top Books by given publisher", booksList)

# %%
#similarity-scores
from sklearn.metrics.pairwise import cosine_similarity
#df_similarity_scores = cosine_similarity(df_pivot_table)

# %%
def collaborative_recommendation(book_name):
    booksList = []
    array_size = np.where(df_pivot_table.index== book_name)[0]
    if array_size.size == 0:
        return create_book_lists_helper("oops! No trending recommendations for the input", booksList)
    book_index = np.where(df_pivot_table.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(df_similarity_scores[book_index])),key=lambda x:x[1],reverse=True)[1:6]
    if len(similar_books) == 0:
        return create_book_lists_helper("Top trending similar books", booksList)
   
    for book in similar_books:
        temp_df = df_books[df_books['Book-Title'] == df_pivot_table.index[book[0]]]
        book_title = temp_df.drop_duplicates('Book-Title')['Book-Title'].values
        book_author = temp_df.drop_duplicates('Book-Title')['Book-Author'].values
        cover_image = temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values
        rBook = Book(book_title[0], cover_image[0], book_author[0])
        booksList.append(rBook)
    
    return create_book_lists_helper("Top trending similar books", booksList)


# %% [markdown]
# ## Books Published Yearly

# %%
# get books published in the same year 
def getBooksYearly(year_or_book: int or str):
    booksList = []
    try:
        year_of_publication = int(year_or_book)
        #valid year checking
        if (year_of_publication < 1900):
            return create_book_lists_helper("oops! Please input the valid year between 1900 - 2022", booksList)
        elif (year_of_publication > 2022):
            return create_book_lists_helper("oops! Please input the valid year between 1900 - 2022", booksList)
        
        #filter books in the same year
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Year-Of-Publication'] == year_of_publication]
    except:
        #check for book name
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Book-Title'].str.lower().str.contains(year_or_book.lower())]
        
        #no books from the same year
        if (len(same_year_books)== 0):
            return create_book_lists_helper("oops! No yearly recommendations for the input", booksList)        
        
        #year of publication of the same book
        year_of_publication = same_year_books.iloc[0]['Year-Of-Publication']
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Year-Of-Publication'] == year_of_publication]

    if (len(same_year_books)== 0):
        return create_book_lists_helper("oops! No recommendations for year input", booksList)

    #top 5 rated books
    same_year_books = same_year_books.sort_values(by="Book-Rating", ascending=False)[:5]
    
    #dropping the duplicates
    same_year_books = same_year_books.drop_duplicates(subset=["Book-Title"])
    for book in same_year_books.values.tolist():
        rBook = Book(book[0], book[4], book[1])
        booksList.append(rBook)
    return create_book_lists_helper("Trending books in the same year", booksList)

# %% [markdown]
# ## Books published at the given place

# %%
#location as input
def getsamePlaceBooks(place):
    booksList = []
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
            return create_book_lists_helper("oops! No recommendations for place input", booksList)
        for book in same_place_books.values.tolist():
            rBook = Book(book[0], book[4], book[1])
            booksList.append(rBook)
        return create_book_lists_helper("Trending books at the same location", booksList)
    else:
        return create_book_lists_helper("oops! No recommendations for place input", booksList)

# %%
#book name as input
def getsamePlaceBooksByTitle(book_name):
    booksList = []
    if book_name is not None:
        book_name = book_name.lower()

        #check for book name
        same_place_books = df_recommendation_dataset[df_recommendation_dataset['Book-Title'].str.lower().str.contains(book_name.lower())]
        
        #no books from the same year
        if (len(same_place_books) == 0):
            return create_book_lists_helper("oops! No recommendations for place input", booksList)
            
    places = ((df_recommendation_dataset['City'].str.lower() == same_place_books.iloc[0]['City'].lower()) |
                  (df_recommendation_dataset['State'].str.lower() == same_place_books.iloc[0]['State'].lower()) |
                  (df_recommendation_dataset['Country'].str.lower() == same_place_books.iloc[0]['Country'].lower()))
    
    if places.any():
        same_place_books = df_recommendation_dataset[places]
        #top 5 rated books
        same_place_books = same_place_books.sort_values(by = "Book-Rating", ascending=False)[:5]
        same_place_books = same_place_books.drop_duplicates(subset=["Book-Title"])
        if(len(same_place_books) == 0):
            return create_book_lists_helper("oops! No recommendations for place input", booksList)    
        for book in same_place_books.values.tolist():
            rBook = Book(book[0], book[4], book[1])
            booksList.append(rBook)
        return create_book_lists_helper("Trending books at the same location", booksList)
    else:
        return create_book_lists_helper("oops! No recommendations for place input", booksList)    

# %% [markdown]
# ### Converting result to JSON format for frontend

# %%
def results_in_json(finalRecommendations):   
    result = json.dumps(finalRecommendations, default=lambda o: o.__dict__, indent=4)
    return result

# %%
# get Final results for all recommendations according to title
def getAllRecommendationsByBookName(name):
    finalRecommendations = []
    results = collaborative_recommendation(name)
    if len(results.books) > 0:
        finalRecommendations.append(collaborative_recommendation(name))
    results = recommendation_by_same_author(name)
    if len(results.books) > 0:
        finalRecommendations.append(recommendation_by_same_author(name))
    results = recommendation_by_same_publisher(name)
    if len(results.books) > 0:
        finalRecommendations.append(recommendation_by_same_publisher(name))
    results = getBooksYearly(name)
    if len(results.books) > 0:
        finalRecommendations.append(getBooksYearly(name))
    results = getsamePlaceBooksByTitle(name)
    if len(results.books) > 0:
        finalRecommendations.append(getsamePlaceBooksByTitle(name))

    if len(finalRecommendations) == 0:
        finalRecommendations.append(create_book_lists_helper("No books found!",[]))
    
    return results_in_json(finalRecommendations)

def getAllRecommendationsByAuthorName(name):
    finalRecommendations = []
    finalRecommendations.append(recommendation_by_given_author(name))
    return results_in_json(finalRecommendations)

def getAllRecommendationsByPublisherName(name):
    finalRecommendations = []
    finalRecommendations.append(recommendation_by_given_publisher(name))
    return results_in_json(finalRecommendations)

def getAllRecommendationsByYear(name):
    finalRecommendations = []
    finalRecommendations.append(getBooksYearly(name))
    return results_in_json(finalRecommendations)

def getAllRecommendationsByLocation(name):
    finalRecommendations = []
    finalRecommendations.append(getsamePlaceBooks(name))
    return results_in_json(finalRecommendations)
