# %%

import numpy as np
import pandas as pd
import seaborn as sns

import pickle

# %%

with open('taxicab.pkl', 'rb') as f: # rb is read binary (wb would be write binary)
    data = pickle.load(f) # f is the file object

len(data)


# %%

states = set(data[0])
for i in range(1, len(data)):
    new_trip = data[i]
    new_states = set(new_trip)
    states = states.union(new_states)

states = list(states)

# %%

# how long is our state space: 38, so we need a 38x38 matrix to track transitions

tr_counts = np.zeros((len(states), len(states)))

# remember our chords exercise (bach.py), you can kindof think of each trip as a song

for trip in data:
    seq = np.array(trip)
    for t in range(1,len(seq)): # starts at 1 because we are doing a markov chain and thus will be back indexing (refering to a previous state)
        # Current and next tokens:
        x_tm1 = seq[t-1] # previous state
        x_t = seq[t] # current state
        # Determine transition indices:
        index_from = states.index(x_tm1)
        index_to = states.index(x_t)
        # Update transition counts:
        tr_counts[index_to, index_from] += 1


# %%

