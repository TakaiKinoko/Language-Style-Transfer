# Language-Style-Transfer
Final Project for Statistical NLP (Fall 19) @NYU


### Set Up
Download **glove.6B.zip** from https://nlp.stanford.edu/projects/glove/. Unzip and and store under **./data** as **./data/glove.6B**


### Project Structure

* ```src/model.py``` runnable file where all components of the model are implemented.
* ```src/importData.ipynb``` jupyter notebook used to demonstrate use of preprocessing functions.
* ```src/preprocessing.py``` functions that preprocess sentences into embeddings.
* ```src/sdLoss.py``` implementation of the style discrepancy loss described in the paper.
