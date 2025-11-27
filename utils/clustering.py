from utils._comment_clustering import (
    clean, 
    get_embeddings, 
    view_elbow_curve,
    assign_clusters,
    plot_clusters,
    get_distances_to_centroid,
    get_cluster_samples
)

import pandas as pd
pd.set_option('display.max_colwidth', 1000)

def compute_embeddings_and_show_elbow_curve(video_id, comments_df):
    df = comments_df[comments_df['isReply'] == 0] # drop all replies
    df = df[df['video_id'] == video_id]
    df = df.reset_index()
    print(f'Embedding {len(df)} comments from video {video_id}')
    df = df.pipe(clean, col='translation')\
        .pipe(get_embeddings)\
        .pipe(view_elbow_curve, max_n=20)
    
    return df


PROMPT_COMMENT_CLUSTERING = """This is a cluster of YouTube video comments. Return a title of max 5 words for this cluster.

    Comments: {}

    Cluster title:"""

def get_prompt_title_clustering(samples):
    samples = [s[:1000] for s in samples]
    samples = [s.replace('"', '') for s in samples]
    samples = [s.replace('\n', '') for s in samples]
    samples = [f'"{s}"' for s in samples]
    samples_string = ', '.join(samples)
    samples_string = samples_string[:10000]
    return PROMPT_COMMENT_CLUSTERING.format(samples_string)

def assign_comments_to_clusters_and_compute_cluster_titles(embeddings_df, num_clusters, gpt_client):
    df, centroids = embeddings_df.pipe(assign_clusters, num_clusters=num_clusters)
    df = df.pipe(plot_clusters, scale_factor=0.12, centroids=centroids)\
        .pipe(get_distances_to_centroid, centroids=centroids)
    # df.to_csv(f'../data/prc/clustering/{processed_video}_cluster_assignment.csv')

    print('===============Cluster Titles===============')
    existing_clusters = df.cluster_assignment.unique().tolist()
    existing_clusters.sort()
    prompt_collection = {}
    for cluster_number in existing_clusters:
        samples = df.pipe(get_cluster_samples, cluster=cluster_number)
        prompt = get_prompt_title_clustering(samples)
        prompt_collection[cluster_number] = prompt
        for temp in [0.2, 0.5, 0.9]:
            rsp = gpt_client.get_response(prompt, temperature=temp, max_tokens=100)
            print(f'{temp}\t{cluster_number}. {rsp}')
        print()

    return df