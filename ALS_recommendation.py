import sys
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random

from sklearn.preprocessing import MinMaxScaler

import implicit # The Cython library

# Load the data like we did before
#raw_data = pd.read_table('data/usersha1-artmbid-artname-plays.tsv')
#raw_data = raw_data.drop(raw_data.columns[1], axis=1)
#raw_data.columns = ['user', 'artist', 'plays']

from datetime import datetime, timedelta



# Drop NaN columns
data = raw_data.dropna()
data = data.copy()

## Create a numeric user_id and artist_id column
#data['user'] = data['user'].astype("category")
#data['artist'] = data['artist'].astype("category")
#data['user_id'] = data['user'].cat.codes
#data['artist_id'] = data['artist'].cat.codes



def create_data(datapath,start_date,end_date):
    df=pd.read_csv(datapath)
    df=df.assign(date=pd.Series(datetime.fromtimestamp(a/1000).date() for a in df.timestamp))
    df=df.sort_values(by='date').reset_index(drop=True) # for some reasons RetailRocket did NOT sort data by date
    df=df[(df.date>=datetime.strptime(start_date,'%Y-%m-%d').date())&(df.date<=datetime.strptime(end_date,'%Y-%m-%d').date())]
    df=df[['visitorid','itemid','event']]
    return df

datapath= 'events.csv'
data=create_data(datapath,'2015-5-3','2015-5-18')
data['event']=data['event'].astype('category')
data['event']=data['event'].cat.codes
# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user) 
# and one for recommendations (user-item)
sparse_item_user = sparse.csr_matrix((data['event'].astype(float), (data['itemid'], data['visitorid'])))
sparse_user_item = sparse.csr_matrix((data['event'].astype(float), (data['itemid'], data['visitorid'])))

# Initialize the als model and fit it using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (sparse_item_user * alpha_val).astype('double')

# Fit the model
model.fit(data_conf)


#---------------------
# FIND SIMILAR ITEMS
#---------------------

# Find the 10 most similar to Jay-Z
item_id = 355908 #Jay-Z
n_similar = 10

# Get the user and item vectors from our trained model
user_vecs = model.user_factors
item_vecs = model.item_factors

# Calculate the vector norms
item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))

# Calculate the similarity score, grab the top N items and
# create a list of item-score tuples of most similar artists

scores = item_vecs.dot(item_vecs[item_id]) / item_norms
scores=np.nan_to_num(scores)
top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
similar = sorted(zip(top_idx, scores[top_idx] / item_norms[item_id]), key=lambda x: -x[1])

# Print the names of our most similar artists
for item in similar:
    idx, score = item
    print(idx,score)

similar = model.similar_items(item_id, n_similar)   

#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

def recommend(user_id, sparse_user_item, user_vecs, item_vecs, num_items=10):
    """The same recommendation function we used before"""

    user_interactions = sparse_user_item[user_id,:].toarray()

    user_interactions = user_interactions.reshape(-1) + 1
    user_interactions[user_interactions > 1] = 0

    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()
    
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    print(user_interactions.shape , rec_vector_scaled.shape)
    recommend_vector = user_interactions * rec_vector_scaled.T

    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    artists = []
    scores = []

    for idx in item_idx:
        artists.append(idx)
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'artist': artists, 'score': scores})

    return recommendations

def recommend_2(user_id, data_sparse, user_vecs, item_vecs, num_items=10):
    """Recommend items for a given user given a trained model
    
    Args:
        user_id (int): The id of the user we want to create recommendations for.
        
        data_sparse (csr_matrix): Our original training data.
        
        user_vecs (csr_matrix): The trained user x features vectors
        
        item_vecs (csr_matrix): The trained item x features vectors
        
        item_lookup (pandas.DataFrame): Used to map artist ids to artist names
        
        num_items (int): How many recommendations we want to return:
        
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items artist names and scores
    
    """
  
    # Get all interactions by the user
    user_interactions = data_sparse[user_id,:].toarray()

    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
    user_interactions[user_interactions > 1] = 0

    # This is where we calculate the recommendation by taking the 
    # dot-product of the user vectors with the item vectors.
    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    # Let's scale our scores between 0 and 1 to make it all easier to interpret.
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions*rec_vector_scaled
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    artists = []
    scores = []

    # Loop through our recommended artist indicies and look up the actial artist name
    for idx in item_idx:
        print(idx)
        print(recommend_vector[idx])
        
        #artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
        scores.append(recommend_vector[idx])

    # Create a new dataframe with recommended artist names and scores
    recommendations = pd.DataFrame({'artist': artists, 'score': scores})
    
    return recommendations
# Get the trained user and item vectors. We convert them to 
# csr matrices to work with our previous recommend function.
user_vecs = sparse.csr_matrix(model.user_factors)
item_vecs = sparse.csr_matrix(model.item_factors)

# Create recommendations for user with id 2025
user_id = 148114#257597
#number of users:
list_scores_id=[]
for i in data['visitorid'].unique().tolist():
    print(i)
    list_scores_id.append([i,model.recommend(user_id, sparse_user_item,N=1)])
list_scores_id=[I[1] for I in list_scores_id]
list_scores_id.sort(key=lambda x: x[1],reverse=True)
    
recommended = model.recommend(user_id, sparse_user_item,N=1221)
recommended.sort(key=lambda x: x[1],reverse=True)

similar_users=model.similar_users(user_id, N=466865)

recommendations = recommend(user_id, sparse_user_item, user_vecs, item_vecs)

print( recommendations)