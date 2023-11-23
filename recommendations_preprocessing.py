## Data Loading

#importing python libraries
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity

#loading datasets
df_books = pd.read_csv('Dataset/Books.csv', low_memory=False)
df_ratings = pd.read_csv('Dataset/Ratings.csv')
df_users = pd.read_csv('Dataset/Users.csv')

#set seed for reproducibility
np.random.seed(0)

## Preprocessing on Books dataset
#first five rows of books dataset
df_books.head()

#number of missing values in books dataset
missing_books_count = df_books.isnull().sum()
missing_books_count

#dropping unrequired columns in books dataset
df_books.drop(['Image-URL-S', 'Image-URL-L'], axis = 1, inplace = True)

#uppercasing ISBN
df_books['ISBN'].str.upper()

#replacing null author and publisher with other
null_Author = np.where(df_books['Book-Author'].isnull())
null_publisher = np.where(df_books['Publisher'].isnull())

df_books.at[null_Author[0][0],'Book-Author'] = 'Other'
df_books.at[null_publisher[0][0],'Publisher'] = 'Other'
df_books.at[null_publisher[0][1],'Publisher'] = 'Other'

#get all the unique values of year of publication
years = df_books['Year-Of-Publication'].unique().sort()

#checking data for 'DK Publishing Inc'
df_books.loc[df_books['Year-Of-Publication'] == 'DK Publishing Inc',:]

#editing data for DK Publishing Inc
df_books.at[209538,'Book-Author'] = 'Other'
df_books.at[209538,'Year-Of-Publication'] = 2000
df_books.at[209538,'Publisher'] = 'DK Publishing Inc'

df_books.at[221678,'Book-Author'] = 'Other'
df_books.at[221678,'Publisher'] = 'DK Publishing Inc'
df_books.at[221678,'Year-Of-Publication'] = 2000

#checking data for 'Gallimard'
df_books.loc[df_books['Year-Of-Publication'] == 'Gallimard',:]

#editing data for Gallimard
df_books.at[220731 ,'Book-Author'] = 'Other'
df_books.at[220731 ,'Publisher'] = 'Gallimard'
df_books.at[220731 ,'Year-Of-Publication'] = '2003'

#converting year of publication in int data type
df_books['Year-Of-Publication'] = df_books['Year-Of-Publication'].astype(int)

#selecting range which less than 2022
df_books.loc[df_books['Year-Of-Publication'] > 2022, 'Year-Of-Publication'] = 2002

#replacing Invalid years with max year
df_books.loc[df_books['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002

#duplicate rows in books dataset
duplicated_books = df_books.duplicated().sum()

# ## Preprocessing on Users dataset
#first five rows of users dataset
df_users.head()

#number of missing values in users dataset
missing_users_count = df_users.isnull().sum()

#splitting location into city, state and country
locations_list = df_users.Location.str.split(', ')
location_count = len(locations_list)
cities_list = []
states_list = []
countries_list = []
for location in range(0, location_count):
    if locations_list[location][0] == '' or locations_list[location][0] == 'n/a' or locations_list[location][0] == ' ':
        cities_list.append('Other')
    else: 
        cities_list.append(locations_list[location][0])

    if (len(locations_list[location]) < 2):
        states_list.append('Other')
        countries_list.append('Other')
    
    else: 
        if locations_list[location][1] == '' or locations_list[location][1] == 'n/a' or locations_list[location][1] == ' ':
            states_list.append('Other')
        else: 
            states_list.append(locations_list[location][1])
        
        if (len(locations_list[location]) < 3):
            countries_list.append('Other')
        
        else: 
            if locations_list[location][2] == '' or locations_list[location][2] == 'n/a' or locations_list[location][2] == ' ':
                countries_list.append('Other')
            else: 
                countries_list.append(locations_list[location][2])

#creating location dataframes
df_city = pd.DataFrame(cities_list, columns=['City'])
df_state = pd.DataFrame(states_list, columns = ['State'])
df_country = pd.DataFrame(countries_list, columns =['Country'])

df_location = pd.concat([df_city, df_state, df_country], axis=1)
df_location

#converting location to lowercase
df_location['City'] = df_location['City'].str.lower()
df_location['State'] = df_location['State'].str.lower()
df_location['Country'] = df_location['Country'].str.lower()

#adding locations to df_users
df_users = pd.concat([df_users, df_location], axis = 1)
df_users

#dropping location from users dataset
df_users.drop(['Location'], axis = 1, inplace = True)

#age preprocessing
ages = df_users['Age'].unique().sort()
considerable_age = df_users[df_users['Age'] <= 98] 
considerable_age = considerable_age[considerable_age['Age'] >= 8]
average_age = round(considerable_age['Age'].mean())

#replacing ages that don't fall in range with average
df_users.loc[df_users['Age'] > 98, 'Age'] = average_age
df_users.loc[df_users['Age'] < 8, 'Age'] = average_age

#filling missing age with average age 
#changing age data type to int
df_users['Age'] = df_users['Age'].fillna(average_age)
df_users['Age'] = df_users['Age'].astype(int)

#duplicate users in books dataset
duplicated_users = df_users.duplicated().sum()
duplicated_users

# ## Preprocessing on Ratings dataset
#first five rows of ratings dataset
df_ratings.head()

#number of missing values in ratings dataset
missing_ratings_count = df_ratings.isnull().sum()
missing_ratings_count

#checking data type of 'Book-Rating'
df_ratings.dtypes

#uppercasing ISBN
df_books['ISBN'].str.upper()

#duplicate ratings in books dataset
duplicated_ratings = df_ratings.duplicated().sum()
duplicated_ratings

# %% [markdown]
# ## Dataset Merging
df_recommendation_dataset = pd.merge(df_books, df_ratings, on="ISBN")
df_recommendation_dataset = pd.merge(df_recommendation_dataset, df_users, on="User-ID")

# %%
df_recommendation_dataset.head()

pickle.dump(df_recommendation_dataset, open('pklFiles/books_with_ratings.pkl', 'wb'))
# %%
#books with ratings
df_books_with_ratings = df_recommendation_dataset[df_recommendation_dataset['Book-Rating'] != 0]
df_books_with_ratings = df_books_with_ratings.reset_index(drop = True)

# %%
#books without ratings
df_books_without_ratings = df_recommendation_dataset[df_recommendation_dataset['Book-Rating'] == 0]
df_books_without_ratings = df_books_without_ratings.reset_index(drop = True)

# ## TOP 50 Books
#calculating total number of ratings for each book
df_ratings_count = df_books_with_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
df_ratings_count = df_ratings_count.sort_values('Book-Rating', ascending=False)

#calculating average ratings 
df_average_rating = df_books_with_ratings.groupby('Book-Title').mean(numeric_only = True)['Book-Rating'].reset_index()
df_average_rating.rename(columns={'Book-Rating':'Average-Rating'},inplace=True)
df_average_rating = df_average_rating.sort_values('Average-Rating', ascending=False)

#merging total-ratings and average-ratings dataset
df_popular_books = pd.merge(df_ratings_count, df_average_rating, on="Book-Title")

#filter to consider total-ratings atleast more than 200
df_top_books = df_popular_books[df_popular_books['Book-Rating']>=200].sort_values('Average-Rating',ascending=False)

# %%
#merge with books for display
df_top_books = df_top_books.merge(df_books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M', 'Book-Rating', 'Average-Rating']]
df_top_books.reset_index(inplace=True)

# %%
def get_top_books():
    top_books = pickle.dump(df_top_books, open('pklFiles/top_books.pkl', 'wb'))
    return top_books
get_top_books()

## Books by same author and publisher

#calculating ratings count on all books
df_total_ratings_count = df_recommendation_dataset.groupby('Book-Title').count()['Book-Rating'].reset_index()
df_total_ratings_count = df_total_ratings_count.sort_values('Book-Rating', ascending=False)

# %%
#calculating average ratings on all books
df_average_books_rating = df_recommendation_dataset.groupby('Book-Title').mean(numeric_only = True)['Book-Rating'].reset_index()
df_average_books_rating.rename(columns={'Book-Rating':'Average-Rating'},inplace=True)

# %%
# merging all the books
df_all_books = df_total_ratings_count.merge(df_average_books_rating,on='Book-Title')

# %%
#calculating aggregared rating
author_recommendations_df = df_all_books.sort_values('Average-Rating', ascending=False)
author_recommendations_df["Aggregated-Rating"] = author_recommendations_df['Book-Rating']*author_recommendations_df['Average-Rating']

# %%
#merging with books
author_recommendations_df = author_recommendations_df.merge(df_books,on='Book-Title').drop_duplicates('Book-Title')
author_recommendations_df = author_recommendations_df.sort_values('Aggregated-Rating',ascending=False)

pickle.dump(author_recommendations_df, open('pklFiles/author_recommendations.pkl', 'wb'))

#### Create objects for the recommended books
class RecommendationList:
    def __init__(self, title, books):
        self.title = title
        self.books = books     
class Book:
    def __init__(self, name, cover, author):
        self.name = name
        self.cover = cover
        self.author = author
#### Helper method to create books in custom list from dataframe

def create_book_lists(title, books):
    recommendation_books = RecommendationList(title, books)
    return recommendation_books

# %% [markdown]
# ## Recommendation for same author

# %%
#Recommend books by same author of the book with book_name as an input  
def recommend_books_by_author(book_name):
    books_list = []
    book_name = book_name.lower()
    book_entry = author_recommendations_df[author_recommendations_df['Book-Title'].str.lower().str.contains(book_name)]
    if book_entry.empty:
        return create_book_lists("oops! No author recommendations for the input", books_list)
    book_author = book_entry['Book-Author'].iloc[0]
    author_recommendations = author_recommendations_df.loc[author_recommendations_df['Book-Author'] == book_author,:][:6]
    author_recommendations.drop(author_recommendations.index[author_recommendations['Book-Title'] == book_name], inplace = True)
    for book in author_recommendations.values.tolist():
        recommended_book = Book(book[0], book[8], book[5])
        books_list.append(recommended_book)
    
    return create_book_lists("Top Books with same author", books_list)
    

# %% [markdown]
# ## Recommendation by the given author name

# %%
def recommendation_by_same_author(author_name):
    books_list = []
    author_name = author_name.lower()
    author_recommendations = author_recommendations_df.loc[author_recommendations_df['Book-Author'].str.lower().str.contains(author_name),:][:5]
    if author_recommendations.empty:
        return create_book_lists("oops! No author recommendations for the input", books_list)
    for book_info in author_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    return create_book_lists("Similar top Books by given author", books_list)

# ## Recommendation for same publisher
#Recommend books by same publisher of the book with book_name as an input  
def recommendation_by_same_publisher(book_name):
    books_list = []
    book_name = book_name.lower()
    book_entry = author_recommendations_df[author_recommendations_df['Book-Title'].str.lower().str.contains(book_name)]
    if book_entry.empty:
        return create_book_lists("oops! No publisher recommendations for the input", books_list)
    book_publisher = book_entry['Publisher'].iloc[0]
    publisher_recommendations = author_recommendations_df.loc[author_recommendations_df['Publisher'] == book_publisher,:][:6]
    publisher_recommendations.drop(publisher_recommendations.index[publisher_recommendations['Book-Title'] == book_name], inplace = True)

    for book_info in publisher_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    return create_book_lists("Top Books published by same publisher", books_list) 

# ## Recommendation by the given publisher name
#recommendation by the given publisher name
def recommendation_by_given_publisher(publisher_name):
    books_list = []
    publisher_name = publisher_name.lower()
    author_recommendations = author_recommendations_df.loc[author_recommendations_df['Publisher'].str.lower().str.contains(publisher_name),:][:5]
    if author_recommendations.empty:
        return create_book_lists("oops! No publisher recommendations for the input", books_list)
    for book_info in author_recommendations.values.tolist():
        recommended_book = Book(book_info[0], book_info[8], book_info[5])
        books_list.append(recommended_book)
    return create_book_lists("Similar top Books by given publisher", books_list)

# ## Collaborative Filtering
#fetching experienced users who have rated at least 200 books
collaborative_user_data = df_recommendation_dataset.groupby('User-ID').count()['Book-Rating'] > 200
experienced_users = collaborative_user_data[collaborative_user_data].index

df_filtered_collaborative_data = df_recommendation_dataset[df_recommendation_dataset['User-ID'].isin(experienced_users)]

#fetching books with minimum 50 ratings by users
collaborative_rating_data = df_filtered_collaborative_data.groupby('Book-Title').count()['Book-Rating'] > 50
books_with_experienced_ratings = collaborative_rating_data[collaborative_rating_data].index 

df_final_collaborative_data = df_filtered_collaborative_data[df_filtered_collaborative_data['Book-Title'].isin(books_with_experienced_ratings)]

#creating pivot table
pivot_table_df = df_final_collaborative_data.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pivot_table_df.fillna(0, inplace=True)

pivot_table_df

#similarity-scores
similarity_scores_df = cosine_similarity(pivot_table_df)

# %%
def collaborative_recommendation(book_name):
    books_list = []
    array_size = np.where(pivot_table_df.index== book_name)[0]
    if array_size.size == 0:
        return create_book_lists("oops! No trending recommendations for the input", books_list)
    book_index = np.where(pivot_table_df.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_scores_df[book_index])),key=lambda x:x[1],reverse=True)[1:5]
    if len(similar_books) == 0:
        return create_book_lists("Top trending similar books", books_list)
   
    for book_info in similar_books:
        temp_df = df_books[df_books['Book-Title'] == pivot_table_df.index[book_info[0]]]
        book_title = temp_df.drop_duplicates('Book-Title')['Book-Title'].values
        book_author = temp_df.drop_duplicates('Book-Title')['Book-Author'].values
        cover_image = temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values
        recommended_book = Book(book_title[0], cover_image[0], book_author[0])
        books_list.append(recommended_book)
    
    return create_book_lists("Top trending similar books", books_list)

# %%
pickle.dump(pivot_table_df, open('pklFiles/pivot_table.pkl', 'wb'))
pickle.dump(similarity_scores_df, open('pklFiles/similarity_scores.pkl', 'wb'))
pickle.dump(df_books,open('pklFiles/books.pkl', 'wb'))

# ## Books Published Yearly
# get books published in the same year 
def recommendations_by_year(year_or_book: int or str):
    books_list = []
    try:
        year_of_publication = int(year_or_book)
        #valid year checking
        if (year_of_publication < 1300):
            return create_book_lists("oops! Please input the valid year between 1900 - 2022", books_list)
        elif (year_of_publication > 2022):
            return create_book_lists("oops! Please input the valid year between 1900 - 2022", books_list)
        
        #filter books in the same year
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Year-Of-Publication'] == year_of_publication]
    except:
        #check for book name
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Book-Title'].str.lower().str.contains(year_or_book.lower())]
        
        #no books from the same year
        if (len(same_year_books)== 0):
            return create_book_lists("oops! No yearly recommendations for the input", books_list)        
        
        #year of publication of the same book
        year_of_publication = same_year_books.iloc[0]['Year-Of-Publication']
        same_year_books = df_recommendation_dataset[df_recommendation_dataset['Year-Of-Publication'] == year_of_publication]

    if (len(same_year_books)== 0):
        return create_book_lists("Trending books in the same year", books_list)

    #top 5 rated books
    same_year_books = same_year_books.sort_values(by="Book-Rating", ascending=False)[:5]
    
    #dropping the duplicates
    same_year_books = same_year_books.drop_duplicates(subset=["Book-Title"])
    
    for book_info in same_year_books.iterrows():
        recommended_book = Book(book_info["Book-Title"], book_info["Image-URL-M"], book_info["Book-Author"])
        books_list.append(recommended_book)
    return create_book_lists("Trending books in the same year", books_list)

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
            return create_book_lists("oops! No recommendations for place input", books_list)
        for book_info in same_place_books.iterrows():
            recommended_book = Book(book_info["Book-Title"], book_info["Image-URL-M"], book_info["Book-Author"])
            books_list.append(recommended_book)
        return create_book_lists("Trending books at the same location", books_list)
    else:
        return create_book_lists("oops! No recommendations for place input", books_list)

#book name as input
def recommendation_by_same_place(book_name):
    books_list = []
    if book_name is not None:
        book_name = book_name.lower()

        #check for book name
        same_place_books = df_recommendation_dataset[df_recommendation_dataset['Book-Title'].str.lower().str.contains(book_name.lower())]
        
        #no books from the same year
        if (len(same_place_books) == 0):
            return create_book_lists("oops! No recommendations for place input", books_list)
            
    places = ((df_recommendation_dataset['City'].str.lower() == same_place_books.iloc[0]['City'].lower()) |
                  (df_recommendation_dataset['State'].str.lower() == same_place_books.iloc[0]['State'].lower()) |
                  (df_recommendation_dataset['Country'].str.lower() == same_place_books.iloc[0]['Country'].lower()))
    
    if places.any():
        same_place_books = df_recommendation_dataset[places]
        #top 5 rated books
        same_place_books = same_place_books.sort_values(by = "Book-Rating", ascending=False)[:5]
        same_place_books = same_place_books.drop_duplicates(subset=["Book-Title"])
        if(len(same_place_books) == 0):
            return create_book_lists("oops! No recommendations for place input", books_list)    
        for book_info in same_place_books.iterrows():
            recommended_book = Book(book_info["Book-Title"], book_info["Image-URL-M"], book_info["Book-Author"])
            books_list.append(recommended_book)
        return create_book_lists("Trending books at the same location", books_list)
    else:
        return create_book_lists("oops! No recommendations for place input", books_list)    

### Converting result to JSON format for frontend
def results_in_json(final_recommendations):   
    result = json.dumps(final_recommendations, default=lambda o: o.__dict__, indent=4)
    return result

# get Final results for all recommendations according to title
def get_recommendations_by_book(book_name):
    final_recommendations = []
    results = collaborative_recommendation(book_name)
    if len(results.books) > 0:
        final_recommendations.append(collaborative_recommendation(book_name))
    results = recommend_books_by_author(book_name)
    if len(results.books) > 0:
        final_recommendations.append(recommend_books_by_author(book_name))
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
        final_recommendations.append(create_book_lists("No books found!",[]))
    
    return results_in_json(final_recommendations)

def get_recommendations_by_author(author_name):
    final_recommendations = []
    final_recommendations.append(recommendation_by_same_author(author_name))
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