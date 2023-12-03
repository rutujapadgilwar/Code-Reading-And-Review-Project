#importing python libraries
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationsPreprocessing:
    def __init__(self):
        self.df_books = self.load_csv('Dataset/Books.csv')
        self.df_ratings = self.load_csv('Dataset/Ratings.csv')
        self.df_users = self.load_csv('Dataset/Users.csv')
        self.df_recommendation_dataset = pd.DataFrame()
        self.df_top_books = pd.DataFrame()
        self.author_recommendations_df = pd.DataFrame()
        self.pivot_table_df = pd.DataFrame()
        self.similarity_scores_df = []

    def load_csv(self,file_path):
        try:
            return pd.read_csv(file_path, low_memory=False)
        except FileNotFoundError as e:
            print(f"Error loading dataset at {file_path}: {e}")
            return None

    #loading datasets
    df_books = load_csv('Dataset/Books.csv')
    df_ratings = load_csv('Dataset/Ratings.csv')
    df_users = load_csv('Dataset/Users.csv')

    def handle_missing_values(df, column, default_value):
        null_indices = np.where(df[column].isnull())[0]
        for index in null_indices:
            df.at[index, column] = default_value

    def save_dataframe_to_pickle(dataframe, filename):
        with open(filename, 'wb') as file:
            pickle.dump(dataframe, file)

    ## Preprocessing on Books dataset

    # Handling Missing Values in 'df_books'
    handle_missing_values(df_books, ['Book-Author', 'Publisher'], 'Other')

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

    #checking data for 'DK Publishing Inc'
    df_books.loc[df_books['Year-Of-Publication'] == 'DK Publishing Inc',:]

    # Editing data for specific cases
    cases_to_edit = [
        (209538, 'Other', 2000, 'DK Publishing Inc'),
        (221678, 'Other', 2000, 'DK Publishing Inc'),
        (220731, 'Other', '2003', 'Gallimard')
    ]

    for case in cases_to_edit:
        index, author_value, year_value, publisher_value = case
        try:
            df_books.at[index, 'Book-Author'] = author_value
            df_books.at[index, 'Year-Of-Publication'] = year_value
            df_books.at[index, 'Publisher'] = publisher_value
        except KeyError as e:
            print(f"Error editing data for index {index}: {e}")

    # Converting year of publication to int and cleaning invalid years
    df_books['Year-Of-Publication'] = pd.to_numeric(df_books['Year-Of-Publication'], errors='coerce')
    df_books.loc[df_books['Year-Of-Publication'] > 2022, 'Year-Of-Publication'] = 2002
    df_books.loc[df_books['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002

    # Splitting location into city, state and country
    df_users[['City', 'State', 'Country']] = df_users['Location'].str.split(', ', expand=True)
    df_users[['City', 'State', 'Country']] = df_users[['City', 'State', 'Country']].applymap(lambda x: 'Other' if pd.isnull(x) or x in ['', 'n/a', ' '] else x.lower())
    df_users.drop(['Location'], axis=1, inplace=True)

    # Age preprocessing
    age_mask = (df_users['Age'] >= 8) & (df_users['Age'] <= 98)
    average_age = round(df_users.loc[age_mask, 'Age'].mean())
    df_users['Age'] = df_users['Age'].fillna(average_age).astype(int)

    # Dataset Merging
    df_recommendation_dataset = pd.merge(df_books, df_ratings, on="ISBN")
    df_recommendation_dataset = pd.merge(df_recommendation_dataset, df_users, on="User-ID")

    save_dataframe_to_pickle(df_recommendation_dataset, 'pklFiles/books_with_ratings.pkl')

    # Books with ratings
    df_books_with_ratings = df_recommendation_dataset[df_recommendation_dataset['Book-Rating'] != 0]
    df_books_with_ratings = df_books_with_ratings.reset_index(drop = True)

    # Calculating total number of ratings for each book
    df_ratings_count = df_books_with_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
    df_ratings_count = df_ratings_count.sort_values('Book-Rating', ascending=False)

    # Calculating average ratings 
    df_average_rating = df_books_with_ratings.groupby('Book-Title').mean(numeric_only = True)['Book-Rating'].reset_index()
    df_average_rating.rename(columns={'Book-Rating':'Average-Rating'},inplace=True)
    df_average_rating = df_average_rating.sort_values('Average-Rating', ascending=False)

    # Merging total-ratings and average-ratings dataset
    df_popular_books = pd.merge(df_ratings_count, df_average_rating, on="Book-Title")

    # Filter to consider total-ratings atleast more than 200
    df_top_books = df_popular_books[df_popular_books['Book-Rating']>=200].sort_values('Average-Rating',ascending=False)

    # Merge with books for display
    df_top_books = df_top_books.merge(df_books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M', 'Book-Rating', 'Average-Rating']]
    df_top_books.reset_index(inplace=True)

    # Top 50 books
    def get_top_books(self):
        top_books = pickle.dump(self.df_top_books, open('pklFiles/top_books.pkl', 'wb'))
        return top_books
    get_top_books()

    # Calculating ratings count on all books
    df_total_ratings_count = df_recommendation_dataset.groupby('Book-Title').count()['Book-Rating'].reset_index()
    df_total_ratings_count = df_total_ratings_count.sort_values('Book-Rating', ascending=False)

    # Calculating average ratings on all books
    df_average_books_rating = df_recommendation_dataset.groupby('Book-Title').mean(numeric_only = True)['Book-Rating'].reset_index()
    df_average_books_rating.rename(columns={'Book-Rating':'Average-Rating'},inplace=True)

    # Merging all the books
    df_all_books = df_total_ratings_count.merge(df_average_books_rating,on='Book-Title')

    # Calculating aggregared rating
    author_recommendations_df = df_all_books.sort_values('Average-Rating', ascending=False)
    author_recommendations_df["Aggregated-Rating"] = author_recommendations_df['Book-Rating']*author_recommendations_df['Average-Rating']

    # Merging with books
    author_recommendations_df = author_recommendations_df.merge(df_books,on='Book-Title').drop_duplicates('Book-Title')
    author_recommendations_df = author_recommendations_df.sort_values('Aggregated-Rating',ascending=False)

    # Fetching experienced users who have rated at least 200 books
    collaborative_user_data = df_recommendation_dataset.groupby('User-ID').count()['Book-Rating'] > 200
    experienced_users = collaborative_user_data[collaborative_user_data].index

    df_filtered_collaborative_data = df_recommendation_dataset[df_recommendation_dataset['User-ID'].isin(experienced_users)]

    # Fetching books with minimum 50 ratings by users
    collaborative_rating_data = df_filtered_collaborative_data.groupby('Book-Title').count()['Book-Rating'] > 50
    books_with_experienced_ratings = collaborative_rating_data[collaborative_rating_data].index 
    df_final_collaborative_data = df_filtered_collaborative_data[df_filtered_collaborative_data['Book-Title'].isin(books_with_experienced_ratings)]

    # Creating pivot table
    pivot_table_df = df_final_collaborative_data.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pivot_table_df.fillna(0, inplace=True)

    similarity_scores_df = cosine_similarity(pivot_table_df)

    save_dataframe_to_pickle(author_recommendations_df, 'pklFiles/author_recommendations.pkl')
    save_dataframe_to_pickle(pivot_table_df, 'pklFiles/pivot_table.pkl')
    save_dataframe_to_pickle(similarity_scores_df, 'pklFiles/similarity_scores.pkl')
    save_dataframe_to_pickle(df_books, 'pklFiles/books.pkl')