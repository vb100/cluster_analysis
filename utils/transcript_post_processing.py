import re
import os
import pandas as pd
import numpy as np
import string
from tqdm import tqdm
from deepmultilingualpunctuation import PunctuationModel

def _remove_double_punctuation(script):
    i = 0
    while i < len(script) - 1:
        if script[i] in string.punctuation:
            if script[i + 1] in string.punctuation:
                script = script[:i] + script[i + 1:]
                continue

        i = i + 1
    return script

def compile_transcript_df_and_correct_punctuation(transcript_folder='../data/prc/transcripts'):
    content_dict = {'video_id' : [], 'content' : []}
    for file in os.listdir(transcript_folder):
        if '.txt' in file:
            with open(os.path.join(transcript_folder, file), 'rt', encoding='utf8') as fin:
                text = fin.read()
                if isinstance(text, str) and len(text) > 0:
                    content_dict['video_id'].append(file[:-4])
                    content_dict['content'].append(text)

    df = pd.DataFrame(content_dict) #.to_csv('../data/prc/transcripts.csv', index=False)
    df = df.reset_index()

    ratios = []
    for text in df['content']:
        text = text.lower().replace('\n', ' ')
        text = re.sub(' +', ' ', text)
        ratios.append(text.count('.') / text.count(' ') if text.count(' ') != 0 else 0)

    to_be_punctuated = [1 if r < (65.3 - 15.2) / 1000 else 0 for r in ratios] # http://www.viviancook.uk/Punctuation/PunctFigs.htm

    print(f'Correcting {np.sum(to_be_punctuated)} transcripts from a total of {len(to_be_punctuated)}')
    assert len(to_be_punctuated) == len(df)

    punctuated_texts = []
    model = PunctuationModel()
    for text, run_punctuation in tqdm(zip(df['content'].tolist(), to_be_punctuated), total=len(df)):
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text)
        
        if run_punctuation == 1:
            text = text.lower()
            if len(text) > 0:
                text =  model.restore_punctuation(text)

        punctuated_texts.append(text)

    punctuated_texts = [_remove_double_punctuation(text) for text in punctuated_texts]
    df['content'] = punctuated_texts
    return df