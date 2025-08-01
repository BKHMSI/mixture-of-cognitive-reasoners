import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine

def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compute_tokenwise_cosine_similarity(array1, array2):
    # array1, array2: [T1, D], [T2, D]
    # Returns: mean of all pairwise cosine similarities between tokens
    similarities = []
    for vec1 in array1:
        for vec2 in array2:
            sim = 1 - cosine(vec1, vec2)
            similarities.append(sim)
    return np.mean(similarities)

def compute_similarity_matrix(emb1_list, emb2_list, num_layers):
    N = len(emb1_list)
    sim_matrix = np.zeros((num_layers, N))

    for layer_idx in range(num_layers):
        for i in range(N):
            emb1 = emb1_list[i][layer_idx]  # shape: [T1, D]
            emb2 = emb2_list[i][layer_idx]  # shape: [T2, D]
            sim = compute_tokenwise_cosine_similarity(emb1, emb2)
            sim_matrix[layer_idx, i] = sim

    return sim_matrix

def plot_heatmap(matrix, expert_pairs, title='Similarity Heatmap'):
    sns.heatmap(matrix, cmap='viridis', square=False, xticklabels=False, yticklabels=True)
    plt.xticks(ticks=np.arange(len(expert_pairs)) + 0.5, labels=[f"{e1} vs {e2}" for e1, e2 in expert_pairs], rotation=45)
    plt.title(title)
    plt.xlabel("Stimulus Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    plt.savefig("figures/similarity_heatmap_layer.png", dpi=300)

# ==== Main ====

expert_pairs = [
    ("language", "social"),
    ("language", "logic"),
    ("language", "world"),
    ("logic", "world"),
    ("logic", "social"),
    ("world", "social"),
]

num_layers = 16  # Number of layers in the model
plot_data = np.zeros((num_layers, len(expert_pairs)))

for idx, (expert_1, expert_2) in enumerate(expert_pairs):
    print(f"Comparing {expert_1} and {expert_2}")

    # Edit these paths:
    file1 = f"outputs/{expert_1}_model_embeddings_baseline_sentences.pkl"
    file2 = f"outputs/{expert_2}_model_embeddings_baseline_sentences.pkl"

    # Load data
    embeddings1 = load_embeddings(file1)  # list of [L, T, D]
    embeddings2 = load_embeddings(file2)

    # Compute NxN similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings1, embeddings2, num_layers)
    plot_data[:, idx] = similarity_matrix.mean(axis=1)

# Plot heatmap
plot_heatmap(plot_data, expert_pairs, title=f"Token-wise Cosine Similarity Across Layers")