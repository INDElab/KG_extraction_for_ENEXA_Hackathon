#a_directory_with_text_files > json_lines_with_chunked_corefed_text.jsonl

import os
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from crosslingual_coreference import Predictor
import argparse



def read_text_files(directory):
    
    text_files = []
    for file in os.listdir(directory):
        file_dict = dict()
        if file.endswith(".txt"):
            file_name = file.split('.')[0].strip()
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                text = f.read()
                file_dict[file_name] = text
        text_files.append(file_dict)
    
    return text_files


def chunk_text(text):
    
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\\', ' ').replace('[citation needed]', ' ').replace('  ', ' ')
    text = sent_tokenize(text)
    chunked_text = list(divide_chunks(text, 2))
    
    return chunked_text


def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]


def get_coref_annotations(text_files, predictor):
    
    data = []
    for ind, file_dict in enumerate(text_files):
        annotations = dict()
        annotations['Document ID'] = ind
        for key, value in file_dict.items():
            annotations['Document Name'] = key
            chunked_text = chunk_text(value)
            for ind, chunk in enumerate(chunked_text):
                annotations['Chunk ID'] = ind
                annotations["Chunk text"] = predictor.predict(chunk)["resolved_text"]
        data.append(annotations)

    return data



def write_jsonl(data, output_dir):
    with open(output_dir, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(f'{line}\n')

    
def main():
    parser = argparse.ArgumentParser(description='Coreference Resolution')
    parser.add_argument('-i', '--input_dir', type=str, default= '/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/downloads/', help='Directory with text files')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/data/', help='Output directory')
    args = parser.parse_args()
    
    predictor = Predictor(language="en_core_web_sm", device=-1, model_name="spanbert")
    text_files = read_text_files(args.input_dir)    
    data = get_coref_annotations(text_files, predictor) 
    write_jsonl(data, args.output_dir)
    
    
    
if __name__ == '__main__':
    main()
        
