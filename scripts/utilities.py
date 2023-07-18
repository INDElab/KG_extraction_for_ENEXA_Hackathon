#utility functions for the project
import requests
import json
import string
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDFS, RDF, OWL, XSD



def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        

def get_data_from_jsonl(filepath):
    """This function reads a jsonl file and returns a list of dictionaries"""
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('None\n')))
            
    return data 


def read_generation_parameters(filepath):
    """This function reads a json file with generation parameters"""
    	
    with open(filepath, 'r', encoding='utf-8') as f:
        params = json.load(f)
        
    return params


# It is a post-processing function to shape the triples from the KnowGL output
def extract_triples(pred):
    """ Just one sequence is passed as argument """
    triples = set()
    entity_set = set()
    entity_triples = set()
    relation_set = set()
    pred = pred.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "") # we remove the brackets and parenthesisstrip() # remove special tokens
        
    if '$' in pred:
        pred = pred.split('$')
        
    else:
        pred = [pred]
            
    pred = [triple.split('|') for triple in pred]   
        
    for triple in pred:
        if len(triple) != 3:
            continue
        if triple[0] == '' or triple[1] == '' or triple[2] == '':
            continue
            
        sbj = triple[0].split('#')
        if len(sbj) != 3:
            continue
        entity_set.add(sbj[0])
        entity_triples.add((sbj[0], "label", sbj[1]))
        entity_triples.add((sbj[0], "type", sbj[2]))
            
        rel = triple[1]
        relation_set.add(rel)
            
        obj = triple[2].split('#')
        if len(obj) != 3:
            continue
        entity_set.add(obj[0])
        entity_triples.add((obj[0], "label", obj[1]))
        entity_triples.add((obj[0], "type", obj[2]))
            
        triples.add((sbj[0], rel, obj[0]))

    return triples, entity_triples, entity_set, relation_set


def write_extractions_to_jsonl(data, filepath):
    """This function writes a list of dictionaries to a jsonl file"""	
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(f'{line}\n')
            

#function to get wikidata IDs
def call_wiki_api(item, item_type='entity'):
  if item_type == 'entity':
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
  if item_type == 'property':
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&type=property&language=en&format=json"
  try:
    data = requests.get(url).json()
    # Return the first id (Could upgrade this in the future)
    return data['search'][0]['id']
  except:
    return 'no-wikiID'


def get_wikidata_id(item_list, item_type):
    """This function returns a dictionary with items as keys and Wikidata IDs as values"""
    
    item_dict = {}
    for item in sorted(item_list):
        item_dict[item] = call_wiki_api(item, item_type)
        
    return item_dict


def write_dict_to_json(dictionary, filepath):
    """This function writes a dictionary to a json file"""
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, indent=4)
  

def get_triple_components(data):
    """This function returns entities, relations, and types
    list of dicts => sets of entities, relations, and types"""
    
    all_entities = set()
    all_relations = set()
    all_types = set()
    
    for line in data:
        entities = line['Extracted Entities']
        all_entities.update(entities)
        
        relations = line['Extracted Relations']
        all_relations.update(relations)
        
        entity_triples = line['Entity Triples and Probabilities']
        for i in entity_triples:
            triple = i[0]
            if triple[1] == 'type':
                ent_type = triple[2]
                all_types.add(ent_type)

    return all_entities, all_relations, all_types


def add_wikidata_triples(data, one_lookup_dict):
    """This function returns maked wikidata triples and adds them to the data """
    
    for line in data:
        extracted_triples = line["Extracted Triples and Probabilities"]
        ent_type_triples = line["Entity Triples and Probabilities"]
        triples_per_line = extracted_triples + ent_type_triples
        wiki_triples = []
        for triple in triples_per_line:
            #print(triple)
            #print(triple[1])
            subj = one_lookup_dict[triple[0][0]]
            if triple[0][1] == 'type':
                rel = 'P31'
            elif triple[0][1] == 'label':
                continue
            else:
                rel = one_lookup_dict[triple[0][1]]
            obj = one_lookup_dict[triple[0][2]]
            wiki_triple = (subj, rel, obj), triple[1]
            #print(wiki_triple)
            wiki_triples.append(wiki_triple)
        #print(wiki_triples)
        line["Wikidata Triples and Probabilities"] = wiki_triples


def get_triples_list(data, target='extracted triples'):
    """This function returns lists of triples
    list of dicts => lists of triples: Extracted triples (as generated by the language model), 
                                       Wikidata triples (corresponding Wikidata IDs of the generated triples), 
                                       Entity type triples (predicate=type or label)
    """

    if target.lower() == 'extracted triples':
        line_key = 'Extracted Triples and Probabilities'
    elif target.lower() == 'wikidata triples':
        line_key = 'Wikidata Triples and Probabilities'
    elif target.lower() == 'entity type triples':
        line_key = 'Entity Triples and Probabilities'

    triple_list = []
    for line in data:
        triples = line[line_key]
        for triple in triples:
           if triple not in triple_list:
               triple_list.append(triple)
               
    return triple_list


def read_wiki_dictionaries(filepath):
    """ read json file with items and their Wikidata IDs
        return a dictionary with items as keys and Wikidata IDs as values
    """ 
    with open(filepath, 'r', encoding='utf-8') as f:
        wiki_dict = json.load(f)
        
    return wiki_dict


def remove_punctuation(text):
    """This function removes punctuation from a text"""
    return text.translate(str.maketrans('', '', string.punctuation))


def shape_relation_name(rel_string, prefix):
    """ 
    Get the generated string as the relation name and shape it to the desired format:
    => desired format: "prefix:relationName"
    """

    rel = remove_punctuation(rel_string)
    rel = rel.strip().lower().split()
    if len(rel) > 1:
        words = []
        for index, word in enumerate(rel):
            if (index % 2) == 0:
                word = word.lower()   
            else:
                word = word.capitalize()
                    
            words.append(word)
        rel = ''.join(words)
    else:
        rel = rel[0].lower()
    shaped_rel =  URIRef(prefix + rel)
    
    return shaped_rel


def shape_entity_name(entity_string, prefix):
    """ 
    Get the generated string as the entity name and shape it to the desired format:
    => desired format: "prefix:entity_name"
    """
    ent = remove_punctuation(entity_string)
    ent = ent.strip().lower().replace(' ', '_')
    ent = URIRef(prefix + ent)
    
    return ent


def shape_class_name(type_string, prefix):
    """ 
    Get the generated string as the class name and shape it to the desired format:
    => desired format: "prefix:ClassName"
    """
    type_name = remove_punctuation(type_string)
    type_name = type_name.strip().split()
    if len(type_name) > 1:
        shaped_type = []
        for x in type_name:
            if x.isupper():
                x=x.upper()
            else:
                x = x.capitalize()
            shaped_type.append(x)
        shaped_type = ''.join(shaped_type)
        shaped_type = URIRef(prefix + shaped_type)  
    else:
        shaped_type = URIRef(prefix + type_name[0].capitalize())
    
    return shaped_type

