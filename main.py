"""
Patent Retrieval API script to host all the endpoints required for Patent Retrieval
Endpoints created:
    - query
Author: Bandi Saideva <bandi.s@outplayhq.com>
Date: 3rd March, 2024
Last Edited By: Bandi Saideva <bandi.s@outplayhq.com>
Date: 3rd March, 2024
"""

# Importing the libraries 
from fastapi import FastAPI, Request
import uvicorn
import os
from pydantic import BaseModel
import sys
import json

# logging
import logging
import json_log_formatter

# model libraries
from sentence_transformers import SentenceTransformer, util
import torch

class PatentRetrievalInput(BaseModel):
    query: str = ""

## Reading Global variables 
SIMILARITY_THRESHOLD = 0.60

# define the formatter and logging file
formatter = json_log_formatter.VerboseJSONFormatter()
json_handler = logging.StreamHandler(sys.stdout)
json_handler.setFormatter(formatter)
# create logger
logger = logging.getLogger('patents_search_logger')
# clearing previous handlers if any
if (logger.handlers):
    logger.handlers.clear()

logger.addHandler(json_handler)
# logging messages are not passed to the handlers of ancestor loggers
logger.propagate = False
# set the level
logger.setLevel(logging.INFO)

# ********************* Loading models *********************************************
def loading_text_similarity_model():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder
    

MODEL = loading_text_similarity_model()
if MODEL == None:
    logger.error("model loading failed") # log


# *********************** helper functions   ***************************************

# function to create list of patents from patents in json.

def read_patent_jsons(directory):
    patents = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            # Read the content of each JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                patents.append(data)

    return patents

# creating embeddings for the abstracts
def create_embeddings(patents):

    corpus = []

    # get the english patents for each patent in patents list of dicts
    for patent in patents:
        if "abstracts" in list(patent.keys()):
            for abstract in patent["abstracts"]:
                if abstract['lang'] == 'EN':
                    corpus.append(abstract["paragraph_markup"])
                    break

    corpus_embeddings = MODEL.encode(corpus, convert_to_tensor=True)

    return corpus_embeddings

patents_dir = "patent_jsons"
PATENTS = read_patent_jsons(patents_dir)
# embeddings of abstracts
CORPUS_EMBEDDINGS = create_embeddings(PATENTS)


# ********************** Creating Endpoints  ****************************************

# Creating the app object
app = FastAPI(title='Patent Retrieval | Patdel Analytics',
              description='An AI-powered feature that helps users to find relevant patent based on the query',
              version='0.1.0')

# add the test hello endpoint
@app.get("/health/status")
def say_hello():
    logger.info("Health check")  # log
    return {"message": "ok"}

# endpoint to return the relevant patents based on query
@app.post("/search")
def patent_retrieval(input: PatentRetrievalInput,request: Request):

    query = input.query

    # Intializing the result with an empty list
    result = []

    # embedding the query
    query_embedding = MODEL.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, CORPUS_EMBEDDINGS)[0]
    top_results = torch.topk(cos_scores, k=10) # k is hyperparameter

    # adding relevant patents to the results based on similarity score
    for score, idx in zip(top_results[0], top_results[1]):

        if score > SIMILARITY_THRESHOLD:
            result.append(PATENTS[idx])
        else:
            break

    return result

if __name__ == "__main__":
    # initiating the app
    uvicorn.run("main:app", host='0.0.0.0', port=8000, workers=1)
