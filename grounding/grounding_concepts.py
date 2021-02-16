'''
Adapted from https://github.com/INK-USC/KagNet/blob/master/grounding/grounding_concepts.py
'''
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


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_"," "))
    lcs = set()
    
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs

def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, None, pattern)
    return matcher

def ground_mentioned_concepts(nlp, matcher, s):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    for match_id, start, end in matches:

        span = doc[start:end].text  # the matched span
        
        original_concept = nlp.vocab.strings[match_id]

        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3] #
        for c in shortest:
            if c in blacklist:
                continue
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect)>0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)


    
    return mentioned_concepts



def match_mentioned_concepts(nlp, tasks, instructions):
    matcher = load_matcher(nlp)

    res = []
    # print("Begin matching concepts.")
    for task, instruction in zip(tasks, instructions):
        
        task_concepts = ground_mentioned_concepts(nlp, matcher, task)
        instruction_concepts = ground_mentioned_concepts(nlp, matcher, instruction)
        
        
        res.append({"task": s, "instruction": a, "tc": list(task_concepts), "ic": list(instruction_concepts)})
    return res

def process(filename):


    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    tasks = []
    instructions = []
    with open(filename, 'r') as f:
        annotations = json.loads(f.read())

    
    for annotation in annotations['turk_annotations']['anns']:
        tasks.append(annotation['task_desc'])
        instructions.append([tok for instr in annotation['high_descs'] for tok in instr]) # flatten instructions


    
    output_path = filename + ".mcp"
    batch_sents = sents
    batch_answers = answers

    res = match_mentioned_concepts(nlp, tasks=tasks, instructions=instructions)
    with open(output_path, 'w') as fo:
        json.dump(res, fo)



def test():
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    res = match_mentioned_concepts(nlp, sents=["Sometimes people say that someone stupid has no swimming pool."], answers=["swimming pool"])
    print(res)


if __name__ == "__main__":
    process(sys.argv[1], int(sys.argv[2]))


