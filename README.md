Attributed Node to Vec
======================

Bring those two things together! Do cool context- and language-aware stuff! Okay, let's fill this a little more precisely now.

Setup
-----

This project uses mostly Julia (with left-over parts of Python in scripts, and more so in notebooks).

For Julia: we're running 1.1, but you can get the [latest version](https://julialang.org/).
To install the required dependencies on your laptop, run `julia environment.jl`.
On a headless server, or a machine on which you don't have the dependencies necessary for [Makie.jl](https://github.com/JuliaPlots/Makie.jl) (notably GLFW), you can instead run `julia environment.headless.jl` to use Makie's Cairo backend.

For Python: using Anaconda, set up the environment with `conda env create -f environment.lock.yml` (this builds the Rust extensions in this package, and installs them locally).

Then activate the anaconda environment, and run `python -m julia.sysimage julia/sys-$(hostname).so` to [build a custom julia system image](https://pyjulia.readthedocs.io/en/stable/sysimage.html#how-to-use-a-custom-system-image), making sure calls from python to julia will work.

Finally, still with the environment activated, run `./setup-datasets.sh` to set up the test datasets in the `datasets/` folder.

### Updating and managing the python environment

`environment.lock.yml` contains the full list of packages and their versions to duplicate the environment we use in development.
To upgrade packages in that environment, use `environment.yml` instead:
it only includes the top-level required packages without versions, and conda will resolve to updated package versions.

`environment.lock-gpu.yml` is a copy of `environment.lock.yml` that uses GPU-enabled TensorFlow.

### Managing large data files

This repository uses [git-annex](https://git-annex.branchable.com/) to manage large data files, which can be a bit complicated. To use this, first make sure you have git-annex installed. If not, and if you don't have root access (e.g. in a HPC environment), Anaconda has it so you can install it inside a dedicated Anaconda environment.

Then, from inside your copy of this repository, you tell git that you're using git-annex:

```bash
git annex init
git annex enableremote warehouse-rsync
```

(Note: the `warehouse-rsync` remote is a so-called "special" annex remote tracked in git, and was created using `git annex initremote warehouse-rsync type=rsync rsyncurl=blondy.lip.ens-lyon.fr:/warehouse/COMPLEXNET/nw2vec/annex encryption=none chunk=10Mib`. You don't need to do that again.)

Now get a copy of all the large files in the `warehouse-rsync` repo:

```bash
git annex get .
```

Now whenever you want to add files to the annex:

```bash
# Commit hashes of the large files
git annex add <the files you want to add>
git commit
# Actually copy your large files to `warehouse-rsync`:
git annex copy --to warehouse-rsync <the files you added>
# Publish your changes upstream (the `--all` makes sure the `git-annex` branch is also pushed)
git push --all
```

If the push operation fails because of a conflict on the `git-annex` branch, you need to merge the remote's version of the `git-annex` branch into yours. You can do that *without having to checkout that branch* by doing:

```bash
git pull
git fetch origin git-annex:git-annex  # Merge what was just pulled from the remote git-annex branch into yours
```

then `git push`ing again should work. Note that since git-annex tracks *which computer has which copy of which file*, whenever someone downloads a large file from the annex things get tracked and the `git-annex` branch is updated. So this `pull/fetch` dance might be necessary quite often as we start working with more people / more computers.

Analyses
--------

You can have a look at the [wiki](https://github.com/ixxi-dante/an2vec/wiki).
The `projects/` folder contains all the notebooks and scripts for the repo's [projects](https://github.com/ixxi-dante/nw2vec/projects), and is probably the most interesting thing to look at.
The `julia/` folder the current implementation of AN2VEC.
The `data/` folder contains any data that may be produced by analyses in `julia/` and `projects/`.

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

