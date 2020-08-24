#=======================================================================
# Author : Pratik Vispute
#=======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



credit = pd.read_csv("tmdb_5000_credits.csv")

movies_df = pd.read_csv("tmdb_5000_movies.csv")

renamed_credit = credit.rename(index = str , columns = {"movie_id" : "id"})

merged_movies = movies_df.merge(renamed_credit , on = "id")

movies = merged_movies.drop(columns = ['homepage','title_x','title_y','status' , 'production_countries'])

#=======================================================================
#Creating variables for applying wighted average method formula
#=======================================================================

v = movies['vote_count']
R = movies['vote_average']
C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.70) #taking 70% of votes 

movies['weighted_average'] = ((R*v)+(C*m)) / (v+m)

movie_rankings = movies.sort_values('weighted_average', ascending = False)
# print(movie_rankings[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20))

weight_avg = movies.sort_values('weighted_average', ascending = False)

# print(weight_avg)




#=======================================================================
# Weighted average plot 
#=======================================================================


# plt.figure(figsize = (12,6))

# axis1 = sns.barplot(x = weight_avg['weighted_average'].head(10),y = weight_avg['original_title'].head(10) , data = weight_avg)

# plt.title('Best Movies by average votes', weight='bold')
# plt.xlabel('Weighted Average Score', weight='bold')
# plt.ylabel('Movie Title', weight='bold')
# plt.savefig('best_movies.png')
# plt.show()

#=======================================================================
# plot based on popularity of the movie
#=======================================================================

# popularity = movies.sort_values('popularity',ascending = False)

# plt.figure(figsize = (12,6))

# axis1 = sns.barplot(x = popularity['popularity'].head(10),y = popularity['original_title'].head(10) , data = popularity)


# plt.title('Best Movies by popularity', weight='bold')
# plt.xlabel('popularity', weight='bold')
# plt.ylabel('Movie Title', weight='bold')
# plt.savefig('best_movies.png')
# plt.show()



#=======================================================================
# giving 50% priority to both features i.e. popularity and weighted average
#=======================================================================

scaling = MinMaxScaler()

scaled_movies = scaling.fit_transform(movies[['weighted_average','popularity']])

normalized_movie = pd.DataFrame(scaled_movies, columns = ['weighted_average','popularity'])

movies[['normalized_weight_average','normalized_popularity']] = normalized_movie

# print(movies.head())

#calculating score for each movie

movies['score'] = movies['normalized_weight_average'] * 0.5 + movies['normalized_popularity'] * 0.5

movies_score = movies.sort_values(['score'],ascending = False)

# print(movies_score[['original_title', 'normalized_weight_average', 'normalized_popularity', 'score']].head(20))




#=======================================================================
# plotting basic recomendation system
#=======================================================================
scored_df = movies.sort_values('score', ascending = False)
plt.figure(figsize=(16,6))

ax = sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df, palette='deep')

#plt.xlim(3.55, 5.25)
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('scored_movies.png')
plt.show()

