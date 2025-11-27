
import pandas as pd
from tqdm import tqdm

MC_TEMPLATE = """This is the transcript of a YouTube video. Which are the two main ideas that the video advocates for? Answer in less than 50 words. Separate the two ideas with a newline.

###
Transcript: "{}"

The video advocates for """

MC_SYSTEM_PROMPT = 'You are a social media analyst.'

def _predict_main_claim(gpt, transcript):
    transcript = transcript.replace('"', '')
    prompt = MC_TEMPLATE.format(transcript[:10_000])
    try:
#         print(prompt)
        response = gpt.get_response(prompt, MC_SYSTEM_PROMPT, temperature=0.5, max_tokens=250, model='gpt-4-turbo-preview')
    except Exception as e:
        print(f'Exception {e}')
        response = 'UNRELATED'
    return response

def _find_first_letter(s):
    return s.find(next(filter(str.isalpha, s)))

def _split_and_format_claims(claims):
    claim_list = []
    for claim in claims.split('\n'):
        claim = claim[_find_first_letter(claim):]
        claim = claim[0].lower() + claim[1:]
        claim_list.append(f'The video advocates for {claim}')
    return claim_list

def get_main_claims_from_transcript(gpt, video_id, transcript_df):
    transcript = transcript_df[transcript_df.video_id == video_id].content.iloc[0]
    claims = _predict_main_claim(gpt, transcript)
    claims = _split_and_format_claims(claims)
    return claims


PS_TEMPLATE = """This is a stance prediction task. Users are making comments about a claim from a YouTube video. Please determine if the comment agrees, disagrees or is unrelated with the claim from the video. If the comment agrees with the claim, answer with AGREE. If the comment disagrees with the claim, answer with DISAGREE. If the comment does not concern the claim, answer with UNRELATED.

###
Claim: {}
Comment: "{}"
Answer: """

PS_SYSTEM_PROMPT = 'You are a bot. You can only answer with "AGREE", "DISAGREE" or "UNRELATED".'

def _predict_stance(gpt, claim, comment):
    claim = claim.replace('"', '')
    comment = comment.replace('"', '')
    prompt = PS_TEMPLATE.format(claim, comment)
    try:
        response = gpt.get_response(prompt, PS_SYSTEM_PROMPT, temperature=0.5, max_tokens=10, model='gpt-4-turbo-preview')
    except Exception as e:
        print(f'Exception {e}')
        response = 'UNRELATED'
    return response


def get_stance_prediction_from_comments(gpt, video_id, claims, comments_df, top_k_comments=None):
    stance_prediction_dict = {'video_id' : [], 'claim' : [], 'comment': [], 'stance' : []}
    tmp_df = comments_df[comments_df['video_id'] == video_id]
    tmp_df['text_len'] = tmp_df.translation.str.len()
    tmp_df = tmp_df.sort_values(by='text_len', ascending=False)
    if top_k_comments:
        tmp_df = tmp_df[:top_k_comments]
    
    for claim in claims:
        for comment in tqdm(tmp_df['translation'], total=len(tmp_df), desc=video_id):
            stance = _predict_stance(gpt, claim, comment)
            stance_prediction_dict['video_id'].append(video_id)
            stance_prediction_dict['claim'].append(claim)
            stance_prediction_dict['comment'].append(comment)
            stance_prediction_dict['stance'].append(stance)

    return pd.DataFrame(stance_prediction_dict)