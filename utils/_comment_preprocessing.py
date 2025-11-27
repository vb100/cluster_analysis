import os
from tqdm import tqdm
import pandas as pd

def compile_comments(df, video_id_col, folder):
    video_id_list = df[video_id_col].tolist()
    comments_df = None
    
    for video_id in tqdm(video_id_list, desc='Stacking video comments'):
        video_dir = os.path.join(folder, video_id)
        if not os.path.isdir(video_dir):
            print(f'Could not found dir {video_dir}')
            continue
            
        for file_name in os.listdir(video_dir):
            if '_comments' in file_name:
                file_path = os.path.join(video_dir, file_name)
                df = pd.read_csv(file_path, index_col=False)
                df['video_id'] = video_id
                if comments_df is not None:
                    comments_df = pd.concat([comments_df, df], ignore_index=True)
                else:
                    comments_df = df
                    
    return comments_df


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[41m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def sanity_check(df, video_df, video_id_col):
    vc = df['video_id'].value_counts()
    successfully_scraped = {i:c for i, c in zip(vc.index.tolist(), vc.tolist())}
    to_scrape = {i:c for i, c in zip(video_df[video_id_col].tolist(), video_df['commentCount'].tolist())}
    per_video_sum = 0
    for k in to_scrape:
        if k not in successfully_scraped:
            print(f'{bcolors.RED}Failed to scrape data for video {k}{bcolors.ENDC}')
            continue
        elif to_scrape[k] != successfully_scraped[k]:
            dif = to_scrape[k] - successfully_scraped[k]
            per_video_sum += dif
            if abs(dif) >= 5:
                color = bcolors.WARNING
            else:
                color = ''
            print(f'{color}Failed to scrape all comments for video {k}: '
                  f'From {to_scrape[k]:5} were returned {successfully_scraped[k]:5}.\t'
                  f'Difference: {dif:5}{bcolors.ENDC}')
            
    print(f'Difference: {per_video_sum}')
#     assert np.sum(video_df['commentCount'].tolist()) - len(comments_df) == per_video_sum
    return df
    

def compute_source_target_columns(df, video_df):
    initial_len = len(df)
    df = df[df['authorName'].notna()]
    df = df[df['authorChannelId'].notna()]
    final_len = len(df)
    print(f'Removed {initial_len - final_len} rows not having source / target data')

    # id = (video_id, channel_id) for video authors and channel_id for others
    id_to_author_name = {(v, c): f'{a} ({v[-5:]})' for v, c, a in zip(video_df['videoId'],
                                                      video_df['channelId'], 
                                                      video_df['channelTitle'])}
    video_id_to_channel_id = {v : c for v, c in id_to_author_name}
    
    for c, a in zip(df['authorChannelId'], df['authorName']):
        id_to_author_name[c] = a # a video author can also comment to a video which does not belong to him
        
    # make author names unique
    existing_author_names = set()
    for k, author in tqdm(id_to_author_name.items(), desc='Making author names unique'):
        if author in existing_author_names:
            i = 2
            new_author_name = f'{author} ({i})'
            while new_author_name in existing_author_names:
                i += 1
                new_author_name = f'{author} ({i})'
                
            id_to_author_name[k] = new_author_name
            existing_author_names.add(new_author_name)
        
        else:
            existing_author_names.add(author)
            
       
    assert len(list(id_to_author_name.values())) == len(set(id_to_author_name.values())), 'Unique test failed'
    
    
    sources = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Computing sources'):
        video_id, channel_id = row['video_id'], row['authorChannelId']
        
        if (video_id, channel_id) in id_to_author_name:
            sources.append(id_to_author_name[(video_id, channel_id)])
        elif channel_id in id_to_author_name:
            sources.append(id_to_author_name[channel_id])
        else:
            raise Exception(f'Source ({video_id}, {channel_id}) could not be found!')
    
    df['Source'] = sources
    
    comment_id_to_channel_id = {com: ch for com, ch in zip(df['id'], df['authorChannelId'])}
    targets = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Computing targets'):
        if row['isReply'] == 1:
            channel_id = comment_id_to_channel_id[row['isReplyTo']]
            targets.append(id_to_author_name[channel_id])
        
        elif row['isReply'] == 0:
            video_id = row['video_id']
            channel_id = video_id_to_channel_id[video_id]
            targets.append(id_to_author_name[(video_id, channel_id)])
            
        else:
            raise Exception(f'Unknown value for isReply')
        video_id, channel_id = row['video_id'], row['authorChannelId']
    
    df['Target'] = targets

    return df