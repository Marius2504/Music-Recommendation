# :calendar: Music Recommandation

## :memo: Description
Music prediction application designed to revolutionize your listening experience, while predicting the next trending songs based on user preferences and current music trends.
Users can discover new music tailored to their taste preferences. Whether you're into pop, rock, hip-hop, or electronic music, it generates a personalized music discovery journey that keeps you engaged and excited.

## :robot: Dataset
### Songs
The dataset in question includes a thorough catalog of the most popular songs from 2023 as listed on Spotify. It offers a wide range of information that goes beyond what is typically available in similar datasets. This dataset provides a wealth of insights into each song's attributes, popularity, and presence on various music platforms. It includes information such as the track name, artist(s) name, release date, Spotify playlists and charts, streaming statistics, Apple Music presence, Deezer presence, Shazam charts, and various audio features.

### Preferences
This dataset consists of a matrix that contains the preferred songs for each user. A song j is preferred by an user i iff prefference[i][j] = 1, otherwise prefference[i][j] = 0

## HPF
Hierarchical Poisson Factorization is a probabilistic approach used for collaborative filtering and matrix factorization tasks. It assumes a Poisson distribution for the observed data.
The implementation consists of 4 gamma-distributed variables that include user activity, preferences, popularity, and song attributes. The observed data, ratings, represent a Poisson distributed variable that combines the user’s preferences with the song’s attributes.

## Hyperparameters 
Starting value: 0.3. Adjusted based on the MSE score.
MCMC steps based on Metropolis. The Metropolis algorithm is a basic form of the more general Metropolis-Hastings algorithm. 

## :camera: Picture
<p align="left">
 <img src="https://github.com/Marius2504/Music-Recommendation/blob/master/predicted_sgs.png" width="600">
</p>

