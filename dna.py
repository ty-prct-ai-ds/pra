import numpy as np
from hmmlearn import hmm
from collections import Counter

nucleotide_map = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

train_sequences = ["ACGTAGCT", "CGTACGTA", 'GATCGTAC']
train_labels = ["GGNNGGNN", "NNGGNNGG", "GGNNGGNN"]

observed_sequences = [
    np.array([nucleotide_map[nuc] for nuc in seq]).reshape(-1, 1) for seq in train_sequences
]

state_mapping = {
    "G": 0,
    "N": 1,
}
state_sequences = [
    np.array([state_mapping[state] for state in label]).reshape(-1, 1) for label in train_labels
]

n_states = 2
n_observations = 4

model = hmm.CategoricalHMM(n_components=n_states,
                           n_iter=100, tol=1e-4, verbose=True)
X_train = np.concatenate(observed_sequences)
lengths = [len(seq) for seq in observed_sequences]

model.fit(X_train, lengths=lengths)

test_sequence = "GTACGTA"
test_observed = np.array([nucleotide_map[nuc]
                         for nuc in test_sequence]).reshape(-1, 1)

predicted_states = model.predict(test_observed)
predicted_lables = ''.join(
    ['G' if s==0 else 'N' for s in predicted_states]
)

print(f"Test sequence: {test_sequence}")
print(f"Predicted labels: {predicted_lables}")
print(f"Predicted states: {predicted_states}")
