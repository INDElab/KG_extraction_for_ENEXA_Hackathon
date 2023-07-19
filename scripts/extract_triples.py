#json_lines_with_chunked_corefed_text.jsonl > kg.ttl

from utilities import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logging.info('Start Logging')


def add_resources_to_KG(KGL, WIKI, g, ent_dict, rel_dict, type_dict):
    """  """
    for key in ent_dict.keys():
        if key == "":
            continue
        else:
            shaped_ent = shape_entity_name(key, KGL)
            
        g.add((shaped_ent, RDF.type, RDFS.Resource))
        g.add((shaped_ent, RDF.type, KGL.Entity))
        g.add((shaped_ent, RDFS.label, Literal(key.strip(), datatype=XSD.string)))
    
    for key, value in ent_dict.items():
        if value == "no-wikiID":
            continue
        else:
            ent = shape_entity_name(key, KGL)
            wiki_ent = URIRef(WIKI + value.strip())
        
        g.add((ent, OWL.sameAs, wiki_ent))
        g.add((wiki_ent, WIKI.id, Literal(value.strip(), datatype=XSD.string)))
        g.add((ent, WIKI.id, Literal(value.strip(), datatype=XSD.string)))
        g.add((wiki_ent, RDF.type, WIKI.Entity))
    ###################################################################################
    
    for key in rel_dict.keys():
        if key == "":
            continue
        else:
            shaped_rel = shape_relation_name(key, KGL)
            
        g.add((shaped_rel, RDF.type, RDF.Property))
        g.add((shaped_rel, RDF.type, KGL.Relation))
        g.add((shaped_rel, RDFS.label, Literal(key.strip(), datatype=XSD.string)))
        
    for key, value in rel_dict.items():
        if value == "no-wikiID":
            continue
        else:
            rel = shape_relation_name(key, KGL)
            wiki_rel = URIRef(WIKI + value.strip())
            
        g.add((rel, OWL.sameAs, wiki_rel))
        g.add((wiki_rel, WIKI.id, Literal(value.strip(), datatype=XSD.string)))
        g.add((rel, WIKI.id, Literal(value.strip(), datatype=XSD.string)))
        g.add((wiki_rel, RDF.type, WIKI.Property))
    ################################################################################################
        
    for key in type_dict.keys():
        if key == "":
            continue
        else:
            shaped_typ = shape_class_name(key, KGL)
            
        g.add((shaped_typ, RDF.type, RDFS.Class))
        g.add((shaped_typ, RDF.type, KGL.Type))
        g.add((shaped_typ, RDFS.label, Literal(key.strip(), datatype=XSD.string)))
    
    for key, value in type_dict.items():
        if value == "no-wikiID":
            continue
        else:
            typ = shape_class_name(key, KGL)
            wiki_typ = URIRef(WIKI + value.strip())
            
        g.add((typ, OWL.sameAs, wiki_typ))
        g.add((wiki_typ, WIKI.id, Literal(value.strip(), datatype=XSD.string)))
        g.add((typ, WIKI.id, Literal(value.strip(), datatype=XSD.string)))
        g.add((wiki_typ, RDF.type, WIKI.Entity))
        g.add((wiki_typ, RDF.type, RDFS.Class))


def add_triple_statements(g, subj, predicate, obj, index, probs, NS): 
    """add triple statements to the graph
    """ 
    triple_node = BNode()
    g.add((triple_node, RDF.type, NS.Triple))
    g.add((triple_node, RDF.type, RDF.Statement))
    g.add((triple_node, NS.tripleID, Literal(index, datatype=XSD.integer)))
    g.add((triple_node, NS.subject, subj))
    g.add((triple_node, NS.predicate, predicate))
    g.add((triple_node, NS.object, obj))
    g.add((triple_node, NS.probability, Literal(probs['prob'], datatype=XSD.float)))
    g.add((triple_node, NS.logProbability, Literal(probs['log_prob'], datatype=XSD.float)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/data/preprocessed_data_for_extraction.jsonl', help='Path to the input file')
    parser.add_argument('--output_folder', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/output', help='Path to the output folder')
    parser.add_argument('--generation_parameters', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/data/generation_parameters.json', help='Path to the generation parameters file')
    parser.add_argument('--KG_folder', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/output/KGs', help='Path to the KG folder')
    parser.add_argument('--gpu', type=str, default='cuda', help = 'whether to use the GPU or not. Default "cuda". Provide "cpu" for running on cpu')
    args = parser.parse_args()
    
    data = get_data_from_jsonl(args.input_file)
    data = data[:5]
    logging.info(f'Number of lines in the input file: {len(data)}')
    
    logging.info(f'Loading the model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained("ibm/knowgl-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("ibm/knowgl-large").to(args.gpu)
    gen_kwargs = read_generation_parameters(args.generation_parameters)

    all_entities = set()
    all_relations = set()
    all_types = set()

    logging.info(f'Extracting triples with the model...')
    for line in data:
        inputs = line["Chunk text"] 
        model_inputs = tokenizer(inputs, max_length=1024, padding=True, truncation=True, return_tensors = 'pt')
        #print(model_inputs['input_ids'].size())
        outputs = model.generate(
                            model_inputs["input_ids"].to(args.gpu),
                            attention_mask=model_inputs["attention_mask"].to(args.gpu),
                            **gen_kwargs,
                            )
        #decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, 
                                                        outputs.beam_indices, normalize_logits=False)
    
        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        output_length = input_length + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = model.generation_config.length_penalty
        reconstructed_scores = transition_scores.cpu().sum(axis=1) / (output_length**length_penalty)
        
        all_triples_dict = dict()
        all_entity_triples_dict = dict()
        entity_set = set()
        relation_set = set()
        type_set = set()
        
        for seq, seq_score, prob in zip(outputs.sequences, reconstructed_scores, np.exp(reconstructed_scores)):
            pred = tokenizer.decode(seq, skip_special_tokens=False)
            #print(f'test:{pred}')
            triples, entity_triples, entities, relations = extract_triples(pred)
            
            triple_dict = dict()
            entity_triple_dict = dict()
            
            for triple in triples:
                triple_dict[triple] = {'prob': float(f'{prob:.2f}'), 'log_prob': float(f'{seq_score:.2f}')}
            all_triples_dict.update(triple_dict)
            
            for entity_triple in entity_triples:
                if entity_triple[1] == 'type':
                    type_set.add(entity_triple[2])
                entity_triple_dict[entity_triple] = {'prob': float(f'{prob:.2f}'), 'log_prob': float(f'{seq_score:.2f}')}
            all_entity_triples_dict.update(entity_triple_dict)
            
            for entity in entities:
                entity_set.add(entity)
            
            for relation in relations:  
                relation_set.add(relation)
                
        all_entities.update(entity_set)
        all_relations.update(relation_set)
        all_types.update(type_set)
        
        line["Extracted Triples and Probabilities"] = list(all_triples_dict.items())
        line["Entity Triples and Probabilities"] = list(all_entity_triples_dict.items())
        line["Extracted Entities"] = list(entity_set)
        line["Extracted Relations"] = list(relation_set)
          
    write_extractions_to_jsonl(data, args.output_folder + '/extractions.jsonl')
    logging.info(f'Extraction completed, outputs are written to {args.output_folder + "/extractions.jsonl"}')
    
    ################################################################################
    logging.info(f'Getting Wikidata IDs for the extracted entities, relations and types...')
    
    rel_dict = get_wikidata_id(all_relations, 'property')
    write_dict_to_json(rel_dict, args.output_folder + '/relations.json')
    logging.info(f'Extraction model extracted {len(rel_dict.keys())} relations')
    logging.info(f'Wikidata IDs for the relations are written to {args.output_folder + "/relations.json"}')
    
    ent_dict = get_wikidata_id(all_entities, 'entity')
    write_dict_to_json(ent_dict, args.output_folder + '/entities.json')
    logging.info(f'Extraction model extracted {len(ent_dict.keys())} entities')
    logging.info(f'Wikidata IDs for the entities are written to {args.output_folder + "/entities.json"}')
    
    type_dict = get_wikidata_id(all_types, 'entity')
    write_dict_to_json(type_dict, args.output_folder + '/types.json')
    logging.info(f'Extraction model extracted {len(type_dict.keys())} types')
    logging.info(f'Wikidata IDs for the entities are written to {args.output_folder + "/types.json"}')
    
    one_lookup_dict = ent_dict.copy()
    one_lookup_dict.update(rel_dict)
    one_lookup_dict.update(type_dict)
    
    ###########################################################################
    
    logging.info(f'Adding Wikidata IDs to the extractions...')
    add_wikidata_triples(data, one_lookup_dict)
    
    write_extractions_to_jsonl(data, args.output_folder + '/extractions_with_wikidata_triples.jsonl')
    logging.info(f'Extractions with Wikidata IDs are written to {args.output_folder + "/extractions_with_wikidata_triples.jsonl"}')
    
    ###########################################################################
    
    logging.info(f'Building the knowledge graph...')
    KGL = Namespace("http://example.org/ibm-KnowGL/#")
    WIKI = Namespace("https://www.wikidata.org/wiki/")

    g = Graph() # create a graph object
    g.bind("knowGL", KGL,  override=True) # bind the knowGL namespace to the graph
    g.bind("wiki", WIKI, override=True) # bind the wiki namespace to the graph

    add_resources_to_KG(KGL, WIKI, g, ent_dict, rel_dict, type_dict)
    
    ###########################################################################
    
    extracted_triples = get_triples_list(data, target='Extracted Triples')
    entity_type_triples = get_triples_list(data, target='Entity Type Triples')
    wiki_triples = get_triples_list(data, target='Wikidata Triples')
    
    ############################################################################
    
    index = 0
    for triple, probs in extracted_triples:
        #print(triple)
        #print(probs)
        #print(probs['prob'])
        index += 1
        #print(index)
        subj = shape_entity_name(triple[0], KGL)
        rel = triple[1]
        predicate = shape_relation_name(rel, KGL)
        if rel == "type":
            obj = shape_class_name(triple[2], KGL)
        elif rel == "subclass":
            obj = shape_class_name(triple[2], KGL)
        elif rel == "label":
            obj = Literal(triple[2].strip(), datatype=XSD.string)
        else:
            obj = shape_entity_name(triple[2], KGL)
        
        shaped_triple = (subj, predicate, obj)
        g.add(shaped_triple)
        add_triple_statements(g, subj, predicate, obj, index, probs, KGL)
        
    for triple, probs in entity_type_triples:
        #print(triple)
        #print(probs)
        #print(probs['prob'])
        index += 1
        #print(index)
        if '' in triple:
            continue
        else:
            subj = shape_entity_name(triple[0], KGL)
            rel = triple[1]
            predicate = shape_relation_name(rel, KGL)
            if rel == "type":
                obj = shape_class_name(triple[2], KGL)
            elif rel == "subclass":
                obj = shape_class_name(triple[2], KGL)
            elif rel == "label":
                obj = Literal(triple[2].strip(), datatype=XSD.string)
            else:
                obj = shape_entity_name(triple[2], KGL)
        
        shaped_triple = (subj, predicate, obj)
        g.add(shaped_triple)
        add_triple_statements(g, subj, predicate, obj, index, probs, KGL)
        
    for triple, probs in wiki_triples:
        #print(triple)
        #print(probs)
        #print(probs['prob'])
        index += 1
        #print(index)
        if '' in triple:
            continue
        else:
            subj = URIRef(WIKI + triple[0].strip())
            predicate = URIRef(WIKI + triple[1].strip())
            obj = URIRef(WIKI + triple[2].strip())
        
        shaped_triple = (subj, predicate, obj)
        g.add(shaped_triple)
    
        add_triple_statements(g, subj, predicate, obj, index, probs, WIKI)
        
    ###########################################################################
    
    g.serialize(destination= args.KG_folder + "/test.ttl", format="turtle")
    logging.info(f'Knowledge graph is written to {args.KG_folder + "/test.ttl"}')
    
if __name__ == '__main__':
    main()
    
    
    