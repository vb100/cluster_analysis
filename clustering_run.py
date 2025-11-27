from utils._comment_clustering import (
        clean,
        get_embeddings,
        view_elbow_curve,
        assign_clusters,
        plot_clusters,
        get_distances_to_centroid,
        get_cluster_samples
    )

from openai import OpenAI
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

import warnings
warnings.filterwarnings("ignore")

PROMPT_COMMENT_CLUSTERING = """You are working as social media data researcher. We are providing a set of comments from YouTube and TikTok from various videos.
Your task: Carefully analyze these comments and return a title of max 6 words for this cluster.
Important: Be specific and neutral.

Expected outcome: Single sentence representing a title of given comments.
"""

def get_prompt_title_clustering(samples):

    api_key = "sk-proj-..."

    client = OpenAI(
        api_key= api_key
        )

    samples = [s[:1000] for s in samples]
    samples = [s.replace('"', '') for s in samples]
    samples = [s.replace('\n', '') for s in samples]
    samples = [f'"{s}"' for s in samples]
    samples_string = ', '.join(samples)
    samples_string = samples_string[:10000]

    chat_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': PROMPT_COMMENT_CLUSTERING},
            {'role': 'user', 'content': samples_string}
        ],
            temperature=0.2,
            max_tokens=100,
            top_p=0.35,
    )
    llm_results = chat_response.choices[0].message.content

    return llm_results


import os
os.environ['OPENAI_API_KEY'] = "sk-proj.."
print(os.environ['OPENAI_API_KEY'])

df_yt = pd.read_csv('SK_results_YT.csv')
df_tt = pd.read_csv('SK_results_TT.csv')
df = pd.concat(objs=[df_yt, df_tt])
df['video_id'] = df['videoId'].astype('str')

# RRPs
# Fiscal
# Regulation
# Political
# Economic_Socio-economic
# Health

TOPIC: str = 'Fiscal'

print(f'Shape before filtering={df.shape}')
df = df[df['theme']==TOPIC]
print(f'Shape after filtering={df.shape}')

from matplotlib import pyplot as plt
import re

def fix_translation(text: str) -> str:
    s = text.replace('!', '! ')
    s = s.replace('  ', ' ')
    s = s.strip()
    return s

def remove_emojis(data):

    data = str(data)
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

    return re.sub(emoj, '', data)

# Remove all emojis from the comments
df['translation_cleaned'] = df['translation'].apply(remove_emojis)
df['translation_cleaned'] = df['translation_cleaned'].apply(fix_translation)

print(f'Translations are cleaned.')

def get_lenght(x: str) -> int:
    return len(x.split(' '))

df['lenght'] = df['translation_cleaned'].apply(get_lenght)
print(df.shape)

_ = plt.hist(
    df['lenght'],
    bins=30
)
plt.show()
print('1 ', df.shape)

# Remove comments which are less than 1 word
df_ = df[df['lenght']>5]
df_ = df_[df_['lenght']<50]
# Leave only regular comments
df_ = df_[df_['kind']=='regular_comment']

print('2 ', df.shape)

_ = plt.hist(
    df_['lenght'],
    bins=30
)
plt.show()

print(df_.shape)


df_clustering = df_.pipe(clean, col='translation_cleaned')\
    .pipe(get_embeddings)\
    .pipe(view_elbow_curve, max_n=20)



df_clustering, centroids = df_clustering.pipe(assign_clusters, num_clusters=8)
print('3 ', df.shape)

df_clustering = df_clustering.pipe(plot_clusters, scale_factor=0.12, centroids=centroids)\
    .pipe(get_distances_to_centroid, centroids=centroids)

for this_cluster in [0, 1, 2, 3, 4, 5, 6, 7]:
    df_temp = df_clustering[df_clustering['cluster_assignment']==this_cluster]['text'].reset_index()
    df_temp.to_csv(f'./clusters/{TOPIC}/{TOPIC}-cluster_{this_cluster}.csv', encoding='utf-8', index=False)
    title: str = get_prompt_title_clustering(samples=list(df_temp['text'].values))
    print(f'Cluster={this_cluster} ({len(df_temp)}) --> {title}')
    print('= '*30)


# segment_4_df = df_clustering[df_clustering['cluster_assignment']==4].drop(columns=['embeddings', 'distance_to_centroid'])
# segment_4_df['text'].value_counts()


# df_[df_['translation'].str.contains('Calin Georgescu chairman', case=False)][
#     [
#         'user_nickname',
#         'translation',
#         'comment_text'
#     ]
# ].sort_values(by=['user_nickname'])
#segment_1_df.sample(4)
