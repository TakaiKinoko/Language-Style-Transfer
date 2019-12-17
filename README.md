# Language-Style-Transfer-with-BERT
Final Project for Statistical NLP (Fall 19) @NYU


### Set Up
Download **glove.6B.zip** from https://nlp.stanford.edu/projects/glove/. Unzip and and store under **./data** as **./data/glove.6B**


### Project Structure

* ```src/model.py``` where all components of the model are implemented at.
* ```src/importData.ipynb``` utility functions to read data from input files.
* ```src/preprocessing.py``` functions that preprocess sentences into embeddings.
* ```src/sdLoss.py``` implementation of the style discrepancy loss described in the paper.