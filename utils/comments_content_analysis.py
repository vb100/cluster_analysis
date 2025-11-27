from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np
# import pickle

def get_triggering_sentences_from_script2(script, comments, similarity_th=0.35):
    nlp = spacy.load("xx_ent_wiki_sm") # python -m spacy download xx_ent_wiki_sm
    nlp.add_pipe('sentencizer')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # script processing
    doc = nlp(script)
    script_sents = [sent.text for sent in doc.sents if len(sent.text) > 30] # have at least 30 letters
    script_sents = [s for s in script_sents if ' ' in s] # have at least two words
    print(f'The script has {len(script_sents)} sentences.')
    script_sents = [' '.join([script_sents[i], script_sents[i+1]]) for i in range(len(script_sents)-1)]
    print(f'The script has {len(script_sents)} 2gram sentences.')
    
    # comment processing
    comments = [c for c in comments if isinstance(c, str)]
    comments = [c for c in comments if len(c) > 20] # have at least 20 letters
    
    # script and comment embeddings
    script_embs = model.encode(script_sents, convert_to_tensor=True, show_progress_bar=True)
    comm_embs = model.encode(comments, convert_to_tensor=True, show_progress_bar=True)
    
    #computing similarity
    sim = util.cos_sim(script_embs, comm_embs)
    most_similar = sim.max(axis=0)
    values_most_similar = most_similar.values
    index_most_similar = most_similar.indices
    
    # removing comments which have low MAX similarity
    index_most_similar[values_most_similar < similarity_th] = -1
    small_sim_count = len(index_most_similar[values_most_similar < similarity_th])
    print(f'Comments with small similarity: {small_sim_count} '
          f'({small_sim_count / len(index_most_similar) * 100:.2f}%)')
    
    
    
    indexes, counts = np.unique(index_most_similar, return_counts=True)
    
    ic = [(i, c) for i, c in zip(indexes, counts)]
    ic.sort(key=lambda x: x[1], reverse=True)
    
    rezult = []
    for i, c in ic:
        if i != -1:
            similar_comments = np.array(comments)[index_most_similar == i]
            rezult.append([script_sents[i], c, similar_comments])
    return rezult


def prepare_comments_content_analysis_data(video_ids, transcript_df, comments_df, similarity_th=0.4):
    video_triggering_comments = {}

    for video_id in video_ids:
        script = transcript_df[transcript_df['video_id'] == video_id]['content'].tolist()[0]
        comms = comments_df[comments_df['video_id'] == video_id]['translation'].tolist()
        triggering_comms = get_triggering_sentences_from_script2(script, comms, similarity_th=similarity_th)            
        video_triggering_comments[video_id] = triggering_comms
        
    # with open('../data/prc/comments_content_trigger.pkl', 'wb') as fout:
    #     pickle.dump(video_triggering_comments, fout)

    return video_triggering_comments