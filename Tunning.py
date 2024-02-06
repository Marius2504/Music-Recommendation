import pymc as pm
import numpy as np

class TuneModel:
    def __init__(self, num_users, num_items, num_components, ratings, parameters):
        self.num_users = num_users
        self.num_items = num_items
        self.parameters = parameters
        self.num_components = num_components
        self.ratings = ratings
        self.parameters = parameters

    def calc_score(self, trace):
        preference_samples = trace['preference']
        attribute_samples = trace['attribute']
        predicted_ratings = np.dot(preference_samples.mean(axis=0), attribute_samples.mean(axis=0).T)
        error = np.square(predicted_ratings - self.ratings).sum()
        score = -error
        return score

    def fineTune(self):
        best_parameters = {}
        best_score = 0
        if self.mode == None: return
        best_score = float("inf")
        for a0 in self.parameters['a0']:
            for b0 in self.parameters['b0']:
                for c0 in self.parameters['c0']:
                    for d0 in self.parameters['d0']:
                        for a in self.parameters['a']:
                            for c in self.parameters['c']:
                                with pm.Model() as hpf_model:
                                    activity_dist = pm.Gamma('activity', a0, a0 / b0, shape=self.self.self.num_users,
                                                             initval=np.ones(self.self.num_users))
                                    preference_dist = pm.Gamma('preference', a, activity_dist[:, np.newaxis],
                                                               shape=(self.num_users, self.num_items),
                                                               initval=np.ones((self.num_users, self.num_items)))
                                    popularity_dist = pm.Gamma('popularity', c0, c0 / d0, shape=self.num_items)
                                    attribute_dist = pm.Gamma('attribute', c, popularity_dist[:, np.newaxis],
                                                              shape=(self.num_items, self.num_components))
                                    ratings_observed = pm.Poisson('ratings',
                                                                  mu=pm.math.dot(preference_dist, attribute_dist),
                                                                  observed=self.ratings)

                                # MCMC
                                with hpf_model:
                                    trace = pm.sample(1000, tune=250, step=pm.Metropolis(), chains=1)

                                current_score = self.calc_score(trace)
                                if current_score < best_score:
                                    best_score = current_score
                                    best_parameters = {'a0': a0, 'b0': b0, 'c0': c0, 'd0': d0, 'a': a, 'c': c}

        return best_parameters, best_score