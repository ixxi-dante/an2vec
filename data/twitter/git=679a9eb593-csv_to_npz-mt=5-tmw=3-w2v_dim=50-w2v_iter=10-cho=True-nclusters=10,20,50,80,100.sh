#!/bin/bash -e

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base36
date
export OUTFOLDER=${0%.sh}
PATH=$PATH:~/bin NUM_WORKERS=40 parallel \
        --bar --header : \
        -j1 \
        --line-buffer \
        --results $OUTFOLDER.csv \
        --env PATH --env NUM_WORKERS --env OUTFOLDER \
        'cd ~/Code/nw2vec && mkdir -p $OUTFOLDER && python projects/twitter/csv_to_npz.py --mutual_threshold 5 --tweet_min_words 3 --w2v_dim 50 --w2v_iter 10 --cluster_hashtags_only {cho} --nclusters {nclusters} --nworkers $NUM_WORKERS --weighted_edgelist_path data/twitter/{dataset}-mention_network.csv --user_tweets_path data/twitter/{dataset}-user_tweets.csv --npz_path $OUTFOLDER/dataset={dataset}-nclusters={nclusters}.npz --cluster2words_path $OUTFOLDER/cluster2words-nclusters={nclusters}.pickle --uid2orig_path $OUTFOLDER/uid2orig-nclusters={nclusters}.pickle --orig2lcc_path $OUTFOLDER/orig2lcc-nclusters={nclusters}.pickle' \
        ::: dataset retweetsrange \
        ::: cho True \
        ::: nclusters 10 20 50 80 100

