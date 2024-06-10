import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import numpy as np
import imageio

# Ensure you have the necessary NLTK resources downloaded

def find_keywords(text, pos_tags=['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
    """
    Find the nouns and verbs in a given text.
    """
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    words = word_tokenize(text)
    tagged = pos_tag(words)
    idx = []
    for i, (word, pos) in enumerate(tagged):
        if word in ['view', 'is']:
            continue
        if pos in pos_tags or word in ['left', 'right', 'top', 'bottom']:
            idx.append(i)
        
    return idx

def export_to_video(video_frames, output_video_path, fps = 24):
    # Ensure all frames are NumPy arrays and determine video dimensions from the first frame
    assert all(isinstance(frame, np.ndarray) for frame in video_frames), "All video frames must be NumPy arrays."
    h, w, _ = video_frames[0].shape

    # Create a video file at the specified path and write frames to it
    with imageio.get_writer(output_video_path, fps=fps, format='mp4') as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path

def split_prompts(prompt):
    
    if " and " in prompt:
        prompt = "|".join(prompt.split(" and "))
    if " on " in prompt:
        phrases = prompt.split(" on ")
        phrases[-1] = "on " + phrases[-1]
        prompt = "|".join(phrases)
    if " in " in prompt:
        phrases = prompt.split(" in ")
        phrases[-1] = "in " + phrases[-1]
        prompt = "|".join(phrases)
    if ", " in prompt:
        prompt = "|".join(prompt.split(", "))
    return prompt
        