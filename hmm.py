import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

np.random.seed(42)

states = ['Sunny', 'Cloudy', 'Rainy']
n_states = len(states)

n_samples = 300

hidden_sequence = np.random.choice(n_states, size=n_samples, p=[0.5, 0.3, 0.2])


def generate_observations(state):
    if state == 0:
        temp = np.random.normal(30, 2)
        humidity = np.random.normal(40, 5)
    elif state == 1:
        temp = np.random.normal(22, 2)
        humidity = np.random.normal(60, 5)
    else:
        temp = np.random.normal(18, 2)
        humidity = np.random.normal(80, 5)

    return [temp, humidity]


observations = np.array([generate_observations(s) for s in hidden_sequence])


discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
obs_discrete = discretizer.fit_transform(observations).astype(int)

obs_combined = obs_discrete[:, 0] * 5 + obs_discrete[:, 1]

model_discrete = hmm.MultinomialHMM(n_components=n_states, n_iter=100)
model_discrete.fit(obs_combined.reshape(-1, 1))

pred_states_discrete = model_discrete.predict(obs_combined.reshape(-1, 1))

acc_discrete = accuracy_score(hidden_sequence, pred_states_discrete)
print(f"\nDiscrete HMM Accuracy: {acc_discrete:.2f}")

model_cont = hmm.GaussianHMM(
    n_components=n_states, n_iter=100, covariance_type='full')
model_cont.fit(observations)
pred_states_cont = model_cont.predict(observations)

acc_cont = accuracy_score(hidden_sequence, pred_states_cont)
print(f"Continuous HMM Accuracy: {acc_cont:.2f}")