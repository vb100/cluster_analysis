import os


def prepare_project_folder_structure():
    try:
        os.mkdir('../data/')
    except FileExistsError:
        print('../data/ already exists.')

    try:
        os.mkdir('../data/raw')
    except FileExistsError:
        print('../data/raw already exists.')

    try:
        os.mkdir('../data/raw/videos')
    except FileExistsError:
        print('../data/raw/videos already exists.')

    try:
        os.mkdir('../data/raw/mp4')
    except FileExistsError:
        print('../data/raw/mp4 already exists.')

    try:
        os.mkdir('../data/raw/users')
    except FileExistsError:
        print('../data/raw/users already exists.')

    try:
        os.mkdir('../data/prc')
    except FileExistsError:
        print('../data/prc already exists.')

    try:
        os.mkdir('../data/prc/summaries')
    except FileExistsError:
        print('../data/prc/summaries already exists.')

    try:
        os.mkdir('../data/prc/thumbnails')
    except FileExistsError:
        print('../data/prc/thumbnails already exists.')

    try:
        os.mkdir('../data/prc/clustering')
    except FileExistsError:
        print('../data/prc/clustering already exists.')

    try:
        os.mkdir('../data/prc/keyword_detection')
    except FileExistsError:
        print('../data/prc/keyword_detection already exists.')

    try:
        os.mkdir('../data/prc/stance_prediction')
    except FileExistsError:
        print('../data/prc/stance_prediction already exists.')