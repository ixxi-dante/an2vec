Node-word-2vec [![Build Status](https://travis-ci.org/ixxi-dante/nw2vec.svg?branch=master)](https://travis-ci.org/ixxi-dante/nw2vec) [![Maintainability](https://api.codeclimate.com/v1/badges/7cff99357c3a27e48768/maintainability)](https://codeclimate.com/github/ixxi-dante/nw2vec/maintainability)
==============

Bring those two things together! Do cool context- and language-aware stuff! Okay, let's fill this a little more precisely now.

Setup
-----

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

### Managing large data files

This repository uses [git-annex](https://git-annex.branchable.com/) to manage large data files, which can be a bit complicated. To use this, first make sure you have git-annex installed. If not, and if you don't have root access (e.g. in a HPC environment), you can install it inside a dedicated Anaconda environment.

Then, from inside your copy of this repository, you tell git that you're using git-annex:

```bash
git annex init
git annex enableremote warehouse-rsync
```

(Note: the `warehouse-rsync` remote is a so-called "special" annex remote tracked in git, and was created using `git annex initremote warehouse-rsync type=rsync rsyncurl=blondy.lip.ens-lyon.fr:/warehouse/COMPLEXNET/nw2vec/annex encryption=none chunk=10Mib`. You don't need to do that again.)

Now whenever you want to add files to the annex:

```bash
git annex add <the files you want to add>
git commit
```

The next step is to actually copy your large files to `warehouse-rsync`:

```bash
git annex copy --to warehouse-rsync <the files you added>
```

Finally, publish your changes upstream:

```
git push --all  # This pushes the `git-annex` branch too, which tracks the hashes of data files
```

If the push operation fails because of a conflict on the `git-annex` branche, you need to merge the remote's version of the `git-annex` branch into yours. You can do that *without having to checkout that branch* by doing:

```bash
git pull
git fetch origin git-annex:git-annex  # Merge what was just pulled from the remote git-annex branch into yours
```

then `git push`ing again should work. Note that since git-annex tracks *which computer has which copy of which file*, whenever someone downloads a large file from the annex things get tracked and the `git-annex` branch is updated. So this `pull/fetch` dance might be necessary quite often as we start working with more people / more computers.

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

