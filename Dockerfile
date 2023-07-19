FROM  python:3.9.6


# get list of installable packets and install wget
RUN apt-get update && \
    apt-get -y install \
		'vim'

# RUN python -m pip install --user requests rdflib transformers torch


COPY data/ /opt/enexa/data/ 
COPY KGs/ /opt/enexa/KGs/
COPY scripts/ /opt/enexa/scripts/

COPY README.md requirements.txt /opt/enexa/

RUN python3 -m venv /opt/enexa/venv/
RUN /bin/bash -c "source /opt/enexa/venv/bin/activate && python3 -m pip install --upgrade pip wheel setuptools && python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r /opt/enexa/requirements.txt"




