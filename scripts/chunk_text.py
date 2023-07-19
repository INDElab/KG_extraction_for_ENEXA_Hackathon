#a_directory_with_text_files > json_lines_with_chunked_corefed_text.jsonl

import os
import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
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


def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]


def chunk_text(text):
    
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\\', ' ').replace('[citation needed]', ' ').replace('  ', ' ')
    text = sent_tokenize(text)
    chunked_text = list(divide_chunks(text, 2))
    
    return chunked_text



def get_annotations(text_files):
    
    data = []
    for ind, file_dict in enumerate(text_files):
        for key, value in file_dict.items():
            annotations = dict()
            annotations['Document ID'] = ind +1 
            annotations['Document Name'] = key
            chunked_text = chunk_text(value)
            for i, chunk in enumerate(chunked_text):
                print(i)
                annotations['Chunk ID'] = i+1
                annotations["Chunk text"] = ''.join(chunk)
                data.append(annotations.copy()) 
                print(annotations)

    return data



def write_jsonl(data, output_dir):
    with open(output_dir + 'preprocessed_data.jsonl', 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(f'{line}\n')

    
def main():
    parser = argparse.ArgumentParser(description='Coreference Resolution')
    parser.add_argument('-i', '--input_dir', type=str, default= '/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/downloads/', help='Directory with text files')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/data/', help='Output directory')
    args = parser.parse_args()
    
    text_files = read_text_files(args.input_dir)    
    data = get_annotations(text_files) 
    write_jsonl(data, args.output_dir)
    print("done chunking!")
    
    
    
if __name__ == '__main__':
    main()
        
