# KG_extraction_for_ENEXA
The repository for the ENEXA Hackathon
This repository contains the following notebooks:
- 1_data_notebook.ipynb
- 2_extraction_notebook_REBEL.ipynb
- 3_get_wikidata_ids_and_triples.ipynb
- 4_build_KG_with_extracted_triples.ipynb
- 5_extraction_notebook_ibm_knowgl-large.ipynb
- 6_build_KG_with_triples_extracted_by_KnowGL.ipynb

We collect and preprocess the data in the first notebook, 1_data_notebook.ipynb. Then, we continue with 2_extraction_notebook_REBEL.ipynb. This notebook is the place where we extract triples from the text. The third notebook is written to get Wikidata IDs. Lastly, a fourth notebook is dedicated to building the actual Knowledge Graph (KG) with the Python rdflib library. In the fifth notebook, we extract triples with IBM-KnowGL model. Then, we build a graph with the extracted triples in the sixth notebook. All resulting data files are stored in the data folder. The final KG files in ttl format can be found in the KGs folder.  
