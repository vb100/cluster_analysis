

from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics import silhouette_score
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import umap.umap_ as umap
from matplotlib import pyplot as plt
from utils.cluster_plot import cluster_plot
from sklearn.metrics.pairwise import euclidean_distances



def clean(df, col):
    data_df = pd.DataFrame({'text' : df[col].tolist()})
    data_df = data_df.dropna()
    data_df = data_df.reset_index()
    return data_df

def get_embeddings(df, from_file=None):
    if from_file:
        with open(from_file, 'rb') as fin:
            embeddings = pickle.load(fin)
            
    else:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        corpus_embeddings = embedder.encode(df['text'], show_progress_bar=True)
        embeddings = {text : embs for text, embs in zip(df['text'], corpus_embeddings)}
    
    corpus_embeddings = [embeddings[s] for s in df['text']]
    df['embeddings'] = corpus_embeddings
    return df


def view_elbow_curve(df, max_n, show_silhouette = True):
    X = np.array(df['embeddings'].tolist())
    print(f'Computing elbow curve for {X.shape[0]} vectors with {X.shape[1]} dimensions...')
    distorsions = []
    sil = []
    
    for k in tqdm(range(2, max_n)):
        kmeans = KMeans(n_clusters=k, max_iter = 10 ** 4, random_state = 42)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)
        # Source: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
        sil.append(silhouette_score(X, kmeans.labels_, metric = 'euclidean')) 

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_n), distorsions, marker='o')
    plt.grid(True)
    plt.title('Elbow curve')
    
    if show_silhouette:
        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(2, max_n), sil, marker='o')
        plt.grid(True)
        plt.title('Silhouette score')
    
    return df

def assign_clusters(df, num_clusters):
    X = np.array(df['embeddings'].tolist())
    kmeans = KMeans(n_clusters=num_clusters, random_state = 42)
    kmeans.fit(X)
    cluster_assignment = kmeans.labels_
    centroids = kmeans.cluster_centers_
    df['cluster_assignment'] = cluster_assignment
    return df, centroids

def plot_clusters(df, scale_factor, centroids):
    """
    ## Cluster plotting
    Code sources:
      * https://stackoverflow.com/questions/23530449/rotate-scale-and-translate-2d-coordinates
      * https://gamedev.stackexchange.com/questions/68891/why-translation-uses-multiplication-and-not-addition
    """
    X = np.array(df['embeddings'].tolist())
    num_clusters = len(df['cluster_assignment'].unique())
    SCALE_FACTOR = scale_factor
    umap_embedder = umap.UMAP(n_neighbors=80,
                            n_components=2,
                            random_state=42,
                            min_dist=0.2,
                            init='random')

    centriods_2d = umap_embedder.fit_transform(centroids)

    umap_embeddings_2d = np.zeros((X.shape[0], 2))

    for i in range(num_clusters):
        mask = df['cluster_assignment']==i
        print(f'Scaling and translating cluster {i} with {X[mask].shape[0]} vectors of {X[mask].shape[1]} dims.')
        cluster_2d = umap_embedder.fit_transform(X[mask])

        # scaling the cluster embeddings
        cluster_2d = cluster_2d * SCALE_FACTOR

        # translating the cluster embeddings
        mean_x, mean_y = cluster_2d.mean(axis=0) # subtract mean in order to center on zero

        cluster_2d = cluster_2d.transpose(1, 0)
        cluster_2d = np.vstack([cluster_2d, np.ones(cluster_2d.shape[1])])

        tx, ty = centriods_2d[i]
        translation_matrix = np.array([[1, 0, tx - mean_x], [0, 1, ty - mean_y], [0, 0, 1]])

        cluster_2d = np.matmul(translation_matrix, cluster_2d)
        cluster_2d = cluster_2d[:2,:].transpose(1,0)

        umap_embeddings_2d[mask] = cluster_2d

    cluster_plot(umap_embeddings_2d,
                 annotations = df['text'].tolist(),
                 cluster_assignment = df['cluster_assignment'],
                 centroids=centriods_2d)
    return df
    
    
def get_distances_to_centroid(df, centroids):
    X = np.array(df['embeddings'].tolist())
    num_clusters = len(df['cluster_assignment'].unique())
    dist = np.zeros((X.shape[0],1))
    for i in range(num_clusters):
            mask = df['cluster_assignment']==i
            dist[mask] = euclidean_distances(X[mask], [centroids[i]])
    df['distance_to_centroid'] = dist
    return df

def get_cluster_samples(df, cluster=0):
    tmp_df = df[df['cluster_assignment'] == cluster]
    tmp_df = tmp_df.sort_values(by='distance_to_centroid')
    return tmp_df['text'].tolist()