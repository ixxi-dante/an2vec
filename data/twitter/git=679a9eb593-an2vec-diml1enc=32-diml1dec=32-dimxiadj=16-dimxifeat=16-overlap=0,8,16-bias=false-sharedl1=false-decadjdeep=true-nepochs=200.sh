#!/bin/bash -e

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base36
date
export OUTFOLDER=${0%.sh}
PATH=$PATH:~/bin OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8 parallel \
        --bar --header : \
        -j5 \
        --env PATH --env JULIA_NUM_THREADS --env OMP_NUM_THREADS --env OUTFOLDER \
        'cd ~/Code/nw2vec && OUTBASE=$OUTFOLDER/dataset={dataset}-nclusters={nclusters}-ld={ld}-dimxi={dimxi} && OUTHISTORY=$OUTBASE-history.npz && OUTWEIGHTS=$OUTBASE-weights.bson && mkdir -p $OUTFOLDER && if test -e $OUTWEIGHTS; then echo "Weights already trained"; false; fi && julia julia/an2vec-dataset.jl --dataset data/twitter/git=679a9eb593-csv_to_npz-mt=5-tmw=3-w2v_dim=50-w2v_iter=10-cho=True-nclusters=10,20,50,80,100/dataset={dataset}-nclusters={nclusters}.npz --label-distribution {ld} --diml1enc 32 --diml1dec 32 --dimxiadj 16 --dimxifeat 16 --overlap $(( 32 - {dimxi} )) --nepochs 200 --bias false --sharedl1 false --decadjdeep {decadjdeep} --savehistory $OUTHISTORY --saveweights $OUTWEIGHTS' \
        ::: decadjdeep true \
        ::: dataset retweetsrange retweetsrange :::+ ld bernoulli normal \
        ::: nclusters 10 20 50 80 100 \
        ::: dimxi $(seq 32 -8 16)
