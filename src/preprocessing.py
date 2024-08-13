'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('model_predictions.csv')
    genres_df = pd.read_csv('genres.csv')
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = genres_df['genre'].unique().tolist()

    # initialize dictionaries to hold counts
    genre_true_counts = {}
    genre_tp_counts = {}
    genre_fp_counts = {}

    for genre in genre_list:
        # true count: number of true instances of each genre
        genre_true_counts[genre] = genres_df[genres_df['genre'] == genre].shape[0]

        # true positive count: model predicts genre correctly
        genre_tp_counts[genre] = ((model_pred_df['predicted_genre'] == genre) & 
                                  (model_pred_df['true_genre'] == genre)).sum()

        # false positive count: model predicts genre but it's incorrect
        genre_fp_counts[genre] = ((model_pred_df['predicted_genre'] == genre) & 
                                  (model_pred_df['true_genre'] != genre)).sum()

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
