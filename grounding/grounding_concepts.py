'''
Adapted from https://github.com/INK-USC/KagNet/blob/master/grounding/grounding_concepts.py
'''
from spacy.language import Language
import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")

def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, [pattern])
    return matcher

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_"," "))
    lcs = set()
    
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs



def ground_mentioned_concepts(nlp, matcher, s):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc, as_spans=True)

    res = []
    span_to_concepts = {}

    for spacy_span in matches:

        match_id = spacy_span.label
        # span = doc[start:end].text  # the matched span
        char_span = (spacy_span.start_char, spacy_span.end_char)
        
        original_concept = nlp.vocab.strings[match_id]

        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]

        if char_span not in span_to_concepts:
            span_to_concepts[char_span] = set()

        span_to_concepts[char_span].add(original_concept)

    # res is a list of dictionary
    for (start_char, end_char), concepts in span_to_concepts.items():
        for concept in concepts:
            res.append({
                'start_char':start_char,
                'end_char':end_char,
                'concept':concept,
                'original_text':s[start_char:end_char]
            })
        

    
    return res



def match_mentioned_concepts(nlp, tasks, instructions, matcher):
    # matcher = load_matcher(nlp)

    res = []
    # print("Begin matching concepts.")
    for task, instructions_ in zip(tasks, instructions):
        
        task_concepts = ground_mentioned_concepts(nlp, matcher, task)
        instruction_concepts = [ground_mentioned_concepts(nlp, matcher, instruction) for instruction in instructions_]
        
        
        res.append({"task": task, "instructions": instructions_, "tc": list(task_concepts), "ic": list(instruction_concepts)})
    return res

def process(filename, nlp=None, matcher=None):

    if nlp is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe('sentencizer')

    if matcher is None:
        matcher = load_matcher(nlp)
    tasks = []
    instructions = []
    with open(filename, 'r') as f:
        annotations = json.loads(f.read())

    
    for annotation in annotations['turk_annotations']['anns']:
        tasks.append(annotation['task_desc'])
        
        instructions.append(annotation['high_descs']) # there are multiple lines of instructions correspond to a goal


    
    output_path = filename + ".mcp"
    

    res = match_mentioned_concepts(nlp, tasks=tasks, instructions=instructions, matcher=matcher)
    
    with open(output_path, 'w') as fo:
        json.dump(res, fo)



def test():
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe('sentencizer')
    res = match_mentioned_concepts(nlp, tasks=["Sometimes people say that someone stupid has no swimming pool."], instructions=["swimming pool"])
    print(res)


if __name__ == "__main__":
    # process(sys.argv[1])
    test()


