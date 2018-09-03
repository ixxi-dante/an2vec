Node-word-2vec [![Build Status](https://travis-ci.org/ixxi-dante/nw2vec.svg?branch=master)](https://travis-ci.org/ixxi-dante/nw2vec) [![Maintainability](https://api.codeclimate.com/v1/badges/7cff99357c3a27e48768/maintainability)](https://codeclimate.com/github/ixxi-dante/nw2vec/maintainability)
==============

Bring those two things together! Do cool context- and language-aware stuff! Okay, let's fill this a little more precisely now.

This repository uses [Git Large File Storage](https://git-lfs.github.com/) (Git-LFS) to store large files. You will have to install Git-LFS to clone the entire contents of the repository.

Setup
-----

TODO: document git-annex setup

First, run `./setup-datasets.sh` to set up the test datasets in the `datasets/` folder.

Next, using Anaconda, set up the environment with `conda env create -f environment.lock.yml` (this builds the Rust extensions in this package, and installs them locally).

### GPU-enabled computations on `grunch`

If you plan on using the LIP `grunch` machine with GPUs, then:

* use `conda env create -f environment.lock-grunchgpu.yml` instead of the above `conda` command,
* run `source load-grunchgpu-modules` to load the required environment modules before running any TensorFlow programs and scripts.

### Updating and managing the environment

`environment.lock.yml` contains the full list of packages and their versions to duplicate the environment we use in development.
To upgrade packages in that environment, use `environment.yml` instead:
it only includes the top-level required packages without versions, and conda will resolve to updated package versions.

`environment.lock-grunchgpu.yml` is a copy of `environment.lock.yml` that uses a GPU-enabled version of TensorFlow 1.6, optimised for LIP's `grunch` machine; it will most probably only work on that machine.

### Optional Rust extension

We started some work on a Rust extension (in `nw2vec-rust`), but it is currently disabled. This will be used in the future to speed up string parsing on the SoSweet dataset.
Now if you want to work on that, install Rust nightly and make it default toolchain in the current directory:

```bash
rustup toolchain install nightly
rustup override set nightly
```

Finally, make sure to run `pip install -e .` whenever you make a change to the extension, to recompile the code and give Python access to it.

Analyses
--------

You can have a look at the [wiki](https://github.com/ixxi-dante/nw2vec/wiki).
The `projects/` folder contains all the notebooks and scripts for the repo's [projects](https://github.com/ixxi-dante/nw2vec/projects), and is probably the most interesting thing to look at.
The `data/` folder contains any data that may be produced by those analyses.

For now, we don't use the Twitter data in the `datasets/` folder, so you can ignore the sections below this and jump directly to the wiki or the projects after having installed the environment.
If, however, you want to play with the Twitter data, read on.

Datasets folders
----------------

After running `./setup-datasets.sh`, the `datasets/` folder contains pointers to all the data we process and the outputs of processing:

```
datasets
├── karate                   # Test networks
├── malariaDBLaNetworks2013  #
├── BlogCatalog-dataset      #
├── Flickr-dataset           #
├── YouTube-dataset          #
├── sosweet-raw              # Raw Twitter data
│     -> /warehouse/COMPLEXNET/TWITTER/data
├── sosweet-network          # Files encoding the network(s) of users and its embedding(s)
│     -> /warehouse/COMPLEXNET/nw2vec/sosweet-network
├── sosweet-text             # Files encoding the text of tweets
│     -> /warehouse/COMPLEXNET/nw2vec/sosweet-text
└── sosweet-w2v              # Files encoding the word2vec embeddings of words from tweets
      -> /warehouse/COMPLEXNET/nw2vec/sosweet-w2v
```

To set these up, and assuming you have access to the raw Twitter data in `datasets/sosweet-raw`:

* Compute the word2vec embeddings: TODO: document
* Extract the networks of users: TODO: document
* Compute the network embeddings by running the `projects/scratchpads/sosweet-node2vec.ipynb` notebook
* Extract the raw text of tweets:
```bash
sosweet-scripts/iter-gz \
    --no-gzip \
    sosweet-scripts/pipe-user_timestamp_body-csv \
    datasets/sosweet-text \
    $(ls datasets/sosweet-raw/2016*.tgz datasets/sosweet-raw/2017*.tgz)
```
* Extract the user ids for the 5-mutual-mention network:
```bash
cat datasets/sosweet-network/undir_weighted_mention_network_thresh_5.txt \
    | sosweet-scripts/pipe-unique_users \
    > datasets/sosweet-network/undir_weighted_mention_network_thresh_5.users.txt
```
* Filter the raw text of tweets for only those users:
```bash
# Create the destination folder
mkdir -p datasets/sosweet-text/undir_weighted_mention_network_thresh_5
# Run the filter
sosweet-scripts/iter-gz \
    --no-gzip \
    "sosweet-scripts/pipe-filter_users datasets/sosweet-network/undir_weighted_mention_network_thresh_5.users.txt" \
    datasets/sosweet-text/undir_weighted_mention_network_thresh_5 \
    $(ls datasets/sosweet-text/*-csv)
```

