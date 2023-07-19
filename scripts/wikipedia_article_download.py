#file_with_a_url_on_each_line.txt > a directory with text files

# Import Modules
from bs4 import *
import requests
import re
import argparse
import json

def get_wiki_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    return urls

def download_wiki_articles(urls, output_dir):
    for url in urls:
        page_name = url.split('/')[-1] # we get the page name from the url
        page = requests.get(url)
        page_content = BeautifulSoup(page.text,'html.parser').select('body')[0]
        page_text = []
        #print(page_content)
        for tag in page_content.find_all(): # we check each tag name
            if tag.name=="p": # For Paragraph we use p tag
                text = tag.text
                text = re.sub(r'\[\d+\]', '', text) # Regex that removes the numbers in square bracets
                text = text.replace('\n',  '').replace('\\', '').replace('[citation needed]', '')
                page_text.append(text)
        page_text = ' '.join(page_text).strip()
        #print(page_text)
        with open(f'{output_dir}{page_name}.txt', 'w', encoding='utf-8') as f:
            f.write(f'{page_text}\n')

def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia Articles')
    parser.add_argument('-f', '--url_file', type=str, default= '/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/urls.json', help='File with a URL on each line')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/finapolat/KG_extraction_for_ENEXA_Hackathon/scripts/downloads/', help='Output directory')
    args = parser.parse_args()
    urls = get_wiki_urls(args.url_file)
    download_wiki_articles(urls, args.output_dir)
    print('Done!')
    
if __name__ == '__main__':
    main()