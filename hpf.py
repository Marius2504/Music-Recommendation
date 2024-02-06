import pymc as pm
import numpy as np

class Model:

    def __init__(self,num_users,num_items,num_components,ratings,a0 = 0.3,b0= 0.3,c0= 0.3,a= 0.3,c= 0.3,d0= 0.3):
        self.trace = None
        self.num_users = num_users
        self.num_items = num_items
        self.num_components = num_components
        self.ratings = ratings
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.a = a
        self.c = c
        self.d0 = d0

    def build(self):
        with pm.Model() as hpf_model:
            # User's activity
            activity_dist = pm.Gamma('activity', self.a0, self.a0 / self.b0, shape=self.um_users, initval=np.ones(self.num_users))

            # User's preferences
            preference_dist = pm.Gamma('preference', self.a, activity_dist[:, np.newaxis], shape=(self.num_users, self.num_items),
                                       initval=np.ones((self.um_users, self.um_items)))

            # Popularity
            popularity_dist = pm.Gamma('popularity', self.c0, self.c0 / self.d0, shape=self.num_items)

            # Song's features
            attribute_dist = pm.Gamma('attribute', self.c, popularity_dist[:, np.newaxis], shape=(self.um_items, self.num_components))

            # Rattings based on Poisson distribution
            ratings_observed = pm.Poisson('ratings', mu=pm.math.dot(preference_dist, attribute_dist), observed=self.ratings)

        # MCMC
        with hpf_model:
            self.trace = pm.sample(1000, tune=250, step=pm.Metropolis(), chains=1)