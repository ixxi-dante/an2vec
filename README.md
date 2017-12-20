Node-word-2vec
==============

Bring those two things together! Do cool context- and language-aware stuff! Okay, let's fill this a little more precisely now.

Data folders
------------

The `data/` folder contains pointers to all the data we process and the outputs of processing:

```
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
```

Setup
-----

First, make sure you have Rust nightly and that it's the default toolchain in the current directory:

```bash
rustup toolchain install nightly
rustup override set nightly
```

Next, using Anaconda, set up the environment with `conda env create -f environment.yml` (this builds the Rust extensions in this package, and installs them locally). If you make changes to the Rust extensions (in `rust-utils`), make sure to run `pip install -e .` to recompile.

Then, in this order, and assuming you have access to the raw Twitter data in `data/sosweet-raw`:

* Compute the word2vec embeddings: TODO: document
* Extract the networks of users: TODO: document
* Compute the network embeddings by running the `sosweet-node2vec.ipynb` notebook
* Extract the raw text of tweets:
```bash
scripts/iter-gz \
    --no-gzip \
    scripts/pipe-user_timestamp_body-csv \
    data/sosweet-text \
    $(ls data/sosweet-raw/2016*.tgz data/sosweet-raw/2017*.tgz)
```
* Extract the user ids for the 5-mutual-mention network:
```bash
cat data/sosweet-network/undir_weighted_mention_network_thresh_5.txt \
    | scripts/pipe-unique_users \
    > data/sosweet-network/undir_weighted_mention_network_thresh_5.users.txt
```
* Filter the raw text of tweets for only those users:
```bash
# Create the destination folder
mkdir -p data/sosweet-text/undir_weighted_mention_network_thresh_5
# Run the filter
scripts/iter-gz \
    --no-gzip \
    "scripts/pipe-filter_users data/sosweet-network/undir_weighted_mention_network_thresh_5.users.txt" \
    data/sosweet-text/undir_weighted_mention_network_thresh_5 \
    $(ls data/sosweet-text/*-csv)
```

Analyses
--------

Nothing done yet.
