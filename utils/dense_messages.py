import spacy
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pandas as pd

def _filter_by_bool(first_list, bool_list):
    filtered_list = [value for value, condition in zip(first_list, bool_list) if condition]
    return filtered_list

def get_dense_messages_in_transcripts(transcript_df, EPS=0.65, MIN_SAMPLES=8):
    sent_df = {'video' : [], 'message' : []}

    nlp = spacy.load("xx_ent_wiki_sm") # python -m spacy download xx_ent_wiki_sm
    nlp.add_pipe('sentencizer')

    for video_id, script in zip(transcript_df['video_id'].tolist(), transcript_df['content'].tolist()):
        doc = nlp(script)
        sents = [sent.text for sent in doc.sents]
        for s in sents:
            if len(s) > 10:
                sent_df['video'].append(video_id)
                sent_df['message'].append(s)
            
    sent_df = pd.DataFrame(sent_df)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    corpus = sent_df['message'].tolist()
    embs = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)


    clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(embs)
    videos = sent_df['video'].tolist()
    print(f'Found {np.max(clustering.labels_) + 1} clusters.')

    clusters = []
    for i in range(np.max(clustering.labels_) + 1):
        msgs = _filter_by_bool(corpus, clustering.labels_ == i)
        vds = _filter_by_bool(videos, clustering.labels_ == i)
        assert len(msgs) == len(vds)
        clusters.append({'messages' : msgs, 'videos' : vds})

    clusters.sort(key=lambda x: len(np.unique(x['videos'])), reverse=True)
    return clusters

def visualize_dense_messages(dense_messages, show_n_samples=2):
    for i, cluster in enumerate(dense_messages):
        print(f"Cluster index: {i}. Unique videos in cluster: {len(np.unique(cluster['videos']))}.")
        try: 
            for i in range(show_n_samples):
                print(f'Cluster sample: \"{cluster["messages"][i]}\"')
        except:
            pass
        print('-' * 80)

def save_dense_messages_for_report(dense_messages, cluster_indexes_to_save, save_dir='../data/prc/keyword_detection'):
    for i in cluster_indexes_to_save:
        pd.DataFrame(dense_messages[i]).to_csv(os.path.join(save_dir, f'keyword_cluster_{i}.csv'))