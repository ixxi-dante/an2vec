Node-word-2vec
==============

Bring those two things together! Do cool context- and language-aware stuff! Okay, let's fill this a little more precisely now.

Data folders
------------

The `data/` folder contains pointers to all the data we process and the outputs of processing:

data
├── karate           # Test network
├── sosweet-raw      # Raw Twitter data
│     -> /datastore/complexnet/twitter/data
├── sosweet-network  # Files encoding the network(s) of users and its embedding(s)
│     -> /datastore/complexnet/nw2vec/sosweet-network
├── sosweet-text     # Files encoding the text of tweets
│     -> /datastore/complexnet/nw2vec/sosweet-text
└── sosweet-w2v      # Files encoding the word2vec embeddings of words from tweets
      -> /datastore/complexnet/nw2vec/sosweet-w2v

Analyses
--------

Setup
^^^^^

First, using Anaconda, set up the environment with `conda env create -f environment.yml`.

Then, in this order, and assuming you have access to the raw Twitter data in `data/sosweet-raw`:
* Compute the word2vec embeddings: TODO: document
* Extract the networks of users: TODO: document
* Compute the network embeddings by running the `sosweet-node2vec.ipynb` notebook
* Extract the raw text of tweets: `scripts/iter-gz scripts/pipe-user_timestamp_body-csv data/sosweet-text $(ls data/sosweet-raw/2016*.tgz data/sosweet-raw/2017*.tgz)`

Actual analyses
---------------

Nothing done yet.
