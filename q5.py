# %%

import numpy as np
import pandas as pd
import seaborn as sns

import pickle


with open('taxicab.pkl', 'rb') as f: # rb is read binary (wb would be write binary)
    data = pickle.load(f) # f is the file object

len(data)


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

print('Transition Counts:\n', tr_counts)
# %%

# sum the transition counts by row:
sums = tr_counts.sum(axis=1, keepdims=True)
print('State Proportions: \n')
print(sums)

# %%

# Normalize the transition count matrix to get proportions:

tr_pr = np.divide(tr_counts, sums, out=np.zeros_like(tr_counts), where=sums!=0)

tr_pr = pd.DataFrame(np.round(tr_pr,2), index=states, columns=states)
print(tr_pr)

# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(tr_pr, 
            cmap='Blues',       # Or 'Blues', 'plasma', whatever looks good
            square=True,          # Keep cells square
            xticklabels=states,
            yticklabels=states,
            cbar_kws={'label': 'Transition Probability'})

plt.title('Transition Probabilities')
plt.xlabel('...To State')
plt.ylabel('From State...')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
