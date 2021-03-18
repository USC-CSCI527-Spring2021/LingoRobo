import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from glob import glob
import networkx as nx
import math
from collections import defaultdict
import os
import spacy

conceptnet_path = '../conceptnet/conceptnet-assertions-5.6.0.csv.en'
splits_data = '../data/splits/oct21.json'
data_path= '../data/json_feat_2.1.0'
shortest_graph_cache = {} ##(source, target) -> [list, of, concept]

def build_conceptnet(cpnet_path):
    df = pd.read_csv(cpnet_path,sep='\t',names=['relation','head','tail','weight'])
    relation_of_interest = ['atlocation','capableof','usedfor','isa','partof']
    df = df.loc[df.relation.isin(relation_of_interest)]
    G = nx.DiGraph()

    for row in df.itertuples():
    #     if row.head in terminal_nodes or row.tail in terminal_nodes:
        G.add_weighted_edges_from([(row.head, row.tail, row.weight)])
        

    return G

def pos_filter(sentence, grounded_result):
    '''
    use spacy pos tag to filter concept that are not NOUN.
    grounded_result
    [{'start_char': 13,
        'end_char': 19,
        'concept': 'sponge',
        'original_text': 'sponge'},
        {'start_char': 29,
        'end_char': 43,
        'concept': 'kitchen_island',
        'original_text': 'kitchen island'},
        {'start_char': 45,
        'end_char': 53,
        'concept': 'place_in',
        'original_text': 'place in'}]
    '''
    
    filtered_result = []
    # check if all the resluts bounded by start_char and end_char are NOUN

    start_char2_pos = {}
    end_char2_pos = {}

    prev_pos = None

    for token in nlp(sentence):
        start_char2_pos[token.idx] = token.pos_
        end_char2_pos[token.idx-1] = prev_pos
        prev_pos = token.pos_
    
    # print(start_char2_pos)
    # print(grounded_result)
    for concept in grounded_result:
        if concept['start_char'] in start_char2_pos and start_char2_pos[concept['start_char']] == 'NOUN' \
        and concept['end_char'] in end_char2_pos and end_char2_pos[concept['end_char']] == 'NOUN':
            filtered_result.append(concept)
            # print(concept['concept'], start_char2_pos[concept['start_char']])


    return filtered_result
    

def construct_shortest_paths(terminal_nodes):
    paths = []
    for i in range(len(terminal_nodes)):
        for j in range(len(terminal_nodes)):
            if i != j:
                try:
                    source_node = terminal_nodes[i]
                    target_node = terminal_nodes[j]

                    if (source_node, target_node) in shortest_graph_cache:
                        s_path = shortest_graph_cache[(source_node, target_node)]
                    else:
                        s_path = nx.algorithms.shortest_paths.generic.shortest_path(G, source=terminal_nodes[i], target=terminal_nodes[j],weight='weight')
                        shortest_graph_cache[(source_node, target_node)] = s_path
                    # print(s_path)
                    paths.append(s_path)
                except Exception as e:
                    
                    continue
    return paths

def process(input_path):
    
    with open(input_path,'r') as f:
        grounded_results = json.loads(f.read())

        # parallel computing to speed up the process
        
        for grounded_result in grounded_results:
            grounded_result['sentence_graphs'] = []
            grounded_result['tc'] = pos_filter(grounded_result['task'], grounded_result['tc'])
            grounded_result['ic'] = [pos_filter(instr, instruction_concept) for instr, instruction_concept in zip(grounded_result['instructions'], grounded_result['ic'])] 
            # print(grounded_result['task'])
            for instruction_concept, instr in zip(grounded_result['ic'], grounded_result['instructions']):
                    
                terminal_nodes = list(set([concept['concept'] for concept in grounded_result['tc'] + instruction_concept ]))
                # print(terminal_nodes, instr)
                shortest_paths =  construct_shortest_paths(terminal_nodes)
                
                for path in shortest_paths:
                    if path not in grounded_result['sentence_graphs']:
                        grounded_result['sentence_graphs'].append(path)

                # print(grounded_result['sentence_graphs'])
    return grounded_results

def process_splits():
    with open(splits_data, 'r') as f:
        splits = json.loads(f.read())

    for k, sub_dir in splits.items():
        
        for task in tqdm(sub_dir):
            # load json file
            json_path = os.path.join(data_path, k, task['task'], 'traj_data.json.mcp')

            results = process(json_path)

            output_path = os.path.join(data_path, k, task['task'], 'traj_data.json.sg')

            with open(output_path, 'w') as fo:
                json.dump(results, fo)

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe('sentencizer')
    G = build_conceptnet(conceptnet_path)
    process_splits()