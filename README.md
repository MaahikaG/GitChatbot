---
title: VersionWise
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Description
This is the code for a Streamlit RAG chatbot hosted on HuggingFace. Its purpose is to help students learn version control using Git.
It uses the mistralai/Mixtral-8x7B-Instruct-v0.1tral LLM model. 
It also connects to a Pinecone vector database, which currently contains vectors from the Git handbook and a basic description of all of the Git commands.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [CI/CD](#ci/cd)

## Installation
This chatbot is publically accessible on HuggingFace Spaces, using the link https://maahikag-versionwise.hf.space. 
If you would like to host it on HuggingFace Spaces yourself, feel free to copy the app.py and requirements.txt from this repository. 
In order to run the chatbot, you would have to get a HuggingFace API token and a Pinecone API token.
You would also have to make a Pinecone vector database with the index name "versionwise" and namespace "git_book". 
You could then use the code in the pinecone_vectors directory to populate your database. 

## Usage
This chatbot is integrated into the website on the VersionWise repository using its public URL. 
This website contains tutorials to teach students about Git branching and version control.
This chatbot is intended to provide students with tailored feedback on their commands. 

## Contributing
The main contribution required for this chatbot involves further populating the Pinecone vector database. 
Currently, the database only contains vectors from the Git handbook, and does not read any information from the VersionWise website. 
A future goal would be to have the chatbot give tailored feedback based on their other actions on the VersionWise website.

## CI/CD
- CD Pipeline
  - Whenever anyone pushes/merges to main, the code within Maahika's HuggingFace Spaces chatbot gets updated.
