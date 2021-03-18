import sys
import json
import random
import grounding_concepts
import os
import spacy
import configparser
from spacy.matcher import Matcher
from tqdm import tqdm
from joblib import Parallel, delayed

splits_data = '../data/splits/oct21.json'
data_path= '../data/json_feat_2.1.0'




def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    filtered_patterns = {}

    # down sample the match patters to speed up processing
    for concept, pattern in all_patterns.items():
        flattened_pattern = tuple((prefix, tok) for word in pattern for prefix, tok in word.items())

        if flattened_pattern not in filtered_patterns:
            filtered_patterns[flattened_pattern] = concept
        # update the match concept if the current one is shorter
        elif len(concept) < len(filtered_patterns[flattened_pattern]):
            filtered_patterns[flattened_pattern] = concept
    

    for flattened_pattern, concept in tqdm(filtered_patterns.items(), desc="Adding patterns to Matcher."):
        
        pattern = [{word[0]:word[1]} for word in flattened_pattern ]
        matcher.add(concept, [pattern])
        
    return matcher

def process_splits():
    with open(splits_data, 'r') as f:
        splits = json.loads(f.read())

    for k, sub_dir in splits.items():
        
        for task in tqdm(sub_dir):
            # load json file
            json_path = os.path.join(data_path, k, task['task'], 'traj_data.json')
            grounding_concepts.process(json_path, nlp, matcher)

        # the below process somehow does not speed up the process on pluslab idk why

        # parallel computing to speed up the process
        # parallel = Parallel(10, backend="threading", verbose=0)
            
        
        # parallel(delayed(grounding_concepts.process)(os.path.join(data_path, k, task['task'], 'traj_data.json'), nlp, matcher) for task in tqdm(sub_dir))


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe('sentencizer')
    matcher = load_matcher(nlp)
    process_splits()
