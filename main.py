import pymc as pm
import graphviz as gv
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from hpf import Model
from Tunning import TuneModel


def load_songs_attributs():
    df = pd.read_csv('/kaggle/input/top-spotify-songs-2023/spotify-2023.csv', encoding="ISO-8859-1")

    scaler = MinMaxScaler()
    le = LabelEncoder()
    df['artist(s)_name'] = le.fit_transform(df['artist(s)_name'])
    df['mode'] = le.fit_transform(df['mode'])
    df['key'] = le.fit_transform(df['key'])
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df = df.dropna(subset=['streams'], axis=0)
    df = df.assign(id=[x for x in range(len(df))])

    for column in df.keys()[1:]:
        scaler = MinMaxScaler()
        df[column] = df[column].astype('string').str.replace(',', '.').astype(float)
        df[column] = scaler.fit_transform([[x] for x in df[column]]).reshape(-1)
    return df, df.values.tolist()[:100]


def loadpref():
    file = open('/kaggle/input/users-generate/preference.txt', 'r')
    lines = []
    for line in file.readlines():
        new = []
        for item in line.split():
            new.append(int(item))
        lines.append(new)

    return lines



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df,songs = load_songs_attributs()
songs.sort(key = lambda x:int(x[8]), reverse = True)

# Input data
num_users = 30
num_items = len(songs)  # 953
num_components = 23
scaler = MinMaxScaler()

activity = np.asarray(scaler.fit_transform([[10, 60, 100, 120, 70, 56, 90, 19, 160, 520,
                                             103, 760, 9100, 1220, 780, 56, 901, 129, 160, 510,
                                             1110, 60, 107, 120, 701, 536, 90, 19, 160, 520]]).reshape(
    -1))  # number of minutes each user listen to

preference = np.asarray(loadpref())
popularity = np.asarray([x[-1] for x in songs])
attribute = []
for idd in popularity:
    for song in songs:
        if song[-1] == idd:
            attribute.append(song[15:-1])

num_components = len(attribute[0])
attribute = np.asarray(attribute)

ratings = preference @ attribute  # scalar product

model = Model(num_users,num_items,num_components,ratings)
model.build()

# Results
pm.summary(model.trace)
pm.plot_trace(model.trace)
pm.plot_posterior(model.trace)


for user in range(30):
    sample = model.trace.posterior['preference'][0][991][user] #991 randomly chosen
    maxi = sample[0]
    summ = 0

    for i in range(250,1000):
        ind = 0
        maxi = -999999999
        sample = model.trace.posterior['preference'][0][i][user]
        for index in range(len(sample)):
            if sample[index] > maxi:
                maxi = sample[index]
                ind = index
        summ +=ind
    print(f"User {user} will probably enjoy "+ songs[int(summ/750)][0])


values = [0.3,0.5,1]
parameters = {'a0': values, 'b0': values, 'c0': values, 'd0': values, 'a': values, 'c': values}
tune = TuneModel(num_users,num_items, num_components, ratings, parameters)
best_parameters, best_score = tune.fineTune()

