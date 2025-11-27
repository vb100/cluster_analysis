import requests
import pandas as pd
import pytubefix
from tqdm import tqdm
import time


def _append_comment_data_to_df(comments_df, content, parent=None):
    for item in content["items"]:
        time.sleep(0.025)
        if not parent:
            legacy_item = {
                "id": item["id"],
                "replyCount": item["snippet"]["totalReplyCount"],
                "likeCount": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
                "publishedAt": item["snippet"]["topLevelComment"]["snippet"][
                    "publishedAt"
                ],
                "authorName": item["snippet"]["topLevelComment"]["snippet"][
                    "authorDisplayName"
                ],
                "text": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
                "authorChannelId": item["snippet"]["topLevelComment"]["snippet"][
                    "authorChannelId"
                ]["value"],
                "isReply": 0,
                "isReplyTo": None,
                "isReplyToName": None,
            }

        else:
            legacy_item = {
                "id": item["id"],
                "replyCount": 0,
                "likeCount": item["snippet"]["likeCount"],
                "publishedAt": item["snippet"]["publishedAt"],
                "authorName": item["snippet"]["authorDisplayName"],
                "text": item["snippet"]["textDisplay"],
                "authorChannelId": item["snippet"]["authorChannelId"]["value"],
                "isReply": 1,  # REPLIES MUST BE EXTRACTED SEPARATELY
                "isReplyTo": parent["authorChannelId"][0],
                "isReplyToName": parent["authorName"][0],
            }

        legacy_item = {k: [v] for k, v in legacy_item.items()}

        print(f"legacy_item={legacy_item}")

        tmp_df = pd.DataFrame(legacy_item)
        if isinstance(comments_df, pd.DataFrame):
            comments_df = pd.concat([comments_df, tmp_df])
        else:
            comments_df = tmp_df

    return comments_df, legacy_item


# Function to get YouTube video comments and replies
def _get_video_comments_and_replies(api_key, video_id):
    print(f"video_id={video_id}")

    # Define the base URL for the YouTube API
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    replies_url = "https://www.googleapis.com/youtube/v3/comments"

    # Set up the parameters for the request
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,  # Max comments per request (YouTube API limit is 100)
        "textFormat": "plainText",
        "key": api_key,
    }

    # Initialize dataframe for storing comments and replies
    comments_df = None

    # Make the initial request
    response = requests.get(base_url, params=params)
    data = response.json()

    # Iterate through all pages of results
    while True:
        # Extract top-level comments
        comments_df, formatted_comment = _append_comment_data_to_df(comments_df, data)

        # For each top-level comment, check if it has replies
        for item in data["items"]:
            if item["snippet"]["totalReplyCount"] > 0:
                # Fetch replies using the comment ID
                comment_id = item["snippet"]["topLevelComment"]["id"]
                replies_params = {
                    "part": "snippet",
                    "parentId": comment_id,
                    "maxResults": 100,  # Max replies per request
                    "textFormat": "plainText",
                    "key": api_key,
                }
                replies_response = requests.get(replies_url, params=replies_params)
                replies_data = replies_response.json()

                # Append replies to the dataframe
                comments_df, _ = _append_comment_data_to_df(
                    comments_df, replies_data, parent=formatted_comment
                )

        # Check if there is a next page token to fetch more comments
        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
            response = requests.get(base_url, params=params)
            data = response.json()
        else:
            break

    return comments_df


def get_comments_for_video_list(videos_to_scrape, api_key):
    comments_list = []

    for video_id in tqdm(videos_to_scrape, desc="Comment scraping status: "):
        print(f"video_id={video_id}")
        tmp_df = _get_video_comments_and_replies(api_key, video_id)
        tmp_df["video_id"] = video_id
        comments_list.append(tmp_df)

    comments_df = pd.concat(comments_list, ignore_index=True)
    return comments_df


def get_mp4_for_video_list(video_id_list, output_dir):
    for video_link in tqdm(video_id_list):
        print(
            f"trying to download video for: https://www.youtube.com/watch?v={video_link}"
        )
        try:
            yt = pytubefix.YouTube(
                f"https://www.youtube.com/watch?v={video_link}",
                use_oauth=True,
                allow_oauth_cache=True,
            )
            video = yt.streams.filter(only_audio=True).first()
            video.download(output_path=output_dir, filename=f"{video_link}.mp4")
        except Exception as e:
            print(f"Video donwload failed for link {video_link}: {e}")


def _get_channel_info(api_key, channel_id):
    """
    Queries the YouTube Data API for information about a YouTube channel.

    Parameters:
    - api_key (str): Your YouTube Data API key.
    - channel_id (str): The ID of the YouTube channel you want to get information about.

    Returns:
    - dict: A dictionary containing information about the channel, or an error message if the request fails.
    """

    # The base URL for querying the YouTube Data API
    url = "https://www.googleapis.com/youtube/v3/channels"

    # Set up the query parameters
    params = {
        "part": "snippet,statistics,contentDetails",
        "id": channel_id,
        "key": api_key,
    }

    # Make the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # If the channel exists, return the channel information
        if "items" in data and len(data["items"]) > 0:
            return data["items"][0]
        else:
            return {"error": "Channel not found or no data available."}
    else:
        return {
            "error": f"Failed to retrieve data. Status code: {response.status_code}"
        }


def get_info_for_channel_list(api_key, channel_list):
    """
    Fetches channel information for a list of YouTube channel IDs and returns a pandas DataFrame.

    Parameters:
    - api_key (str): Your YouTube Data API key.
    - channel_list (list): A list of YouTube channel IDs.

    Returns:
    - pd.DataFrame: A pandas DataFrame with channel information (id, title, description, customUrl, publishedAt, viewCount, subscriberCount, videoCount).
    """

    # List to hold the results
    channel_data = []

    # Loop through each channel ID and fetch information
    for channel_id in channel_list:
        channel_info = _get_channel_info(api_key, channel_id)

        snippet = channel_info["snippet"]
        statistics = channel_info["statistics"]

        # Add channel details to the list
        channel_data.append(
            {
                "id": channel_info["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "customUrl": snippet.get("customUrl", ""),
                "publishedAt": snippet.get("publishedAt", ""),
                "viewCount": statistics.get("viewCount", 0),
                "subscriberCount": statistics.get("subscriberCount", 0),
                "videoCount": statistics.get("videoCount", 0),
                "country": snippet.get("country", ""),
            }
        )

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(
        channel_data,
        columns=[
            "id",
            "title",
            "description",
            "customUrl",
            "publishedAt",
            "viewCount",
            "subscriberCount",
            "videoCount",
            "country",
        ],
    )

    return df


def _get_video_info(api_key, video_id):
    """
    Get information about a YouTube video using the YouTube Data API.

    Parameters:
    - api_key (str): Your YouTube Data API key.
    - video_id (str): The YouTube video ID.

    Returns:
    - dict: A dictionary containing the video information.
    """

    # Define the endpoint URL
    url = "https://www.googleapis.com/youtube/v3/videos"

    # Define the parameters for the API request
    params = {
        "part": "snippet,contentDetails,statistics",  # You can specify what parts of the video info you want
        "id": video_id,  # The video ID to fetch information for
        "key": api_key,  # Your YouTube Data API key
    }

    # Make the GET request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()
        # Return the video information (or an empty dict if no results found)
        if "items" in data and len(data["items"]) > 0:
            return data["items"][0]
        else:
            return {"error": "No video found for the given ID"}
    else:
        # Return an error if the request failed
        return {
            "error": f"Failed to retrieve video information, status code: {response.status_code}"
        }


def get_video_metadata(api_key, video_ids):
    """
    Get information for a list of YouTube videos and return a pandas DataFrame.

    Parameters:
    - api_key (str): Your YouTube Data API key.
    - video_ids (list of str): List of YouTube video IDs.

    Returns:
    - pd.DataFrame: A pandas DataFrame with video information including channelId,
                    title, description, channelTitle, viewCount, likeCount, commentCount.
    """
    # Define an empty list to store the data
    video_data = []

    # Loop through the list of video IDs
    for video_id in video_ids:
        # Get the video info using the _get_video_info function
        video_info = _get_video_info(api_key, video_id)

        # Check if there was an error or no video found
        if "error" in video_info:
            print(f"Error retrieving video {video_id}: {video_info['error']}")
            continue

        # Extract the required fields (handling missing data with .get())
        channel_id = video_info["snippet"].get("channelId")
        title = video_info["snippet"].get("title")
        description = video_info["snippet"].get("description")
        channel_title = video_info["snippet"].get("channelTitle")
        published_at = video_info["snippet"].get("publishedAt")
        view_count = video_info["statistics"].get("viewCount", 0)
        like_count = video_info["statistics"].get("likeCount", 0)
        comment_count = video_info["statistics"].get("commentCount", 0)

        # Append the extracted data as a dictionary
        video_data.append(
            {
                "channelId": channel_id,
                "title": title,
                "description": description,
                "channelTitle": channel_title,
                "published": published_at,
                "viewCount": view_count,
                "likeCount": like_count,
                "commentCount": comment_count,
                "videoId": video_id,
            }
        )

    # Create a DataFrame from the list of video data
    df = pd.DataFrame(
        video_data,
        columns=[
            "videoId",
            "channelId",
            "title",
            "description",
            "channelTitle",
            "published",
            "viewCount",
            "likeCount",
            "commentCount",
        ],
    )

    return df
