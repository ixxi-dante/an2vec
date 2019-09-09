#!/bin/bash -e

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base36
date
#        --sshloginfile parallel-slaves-graphsage --sshdelay 0.1 \
PATH=$PATH:~/bin OMP_NUM_THREADS=12 JULIA_NUM_THREADS=12 parallel \
        --bar --header : \
        -j6 \
        --results data/behaviour/gae-benchmarks/graphsage-git=f31b6c5296-dataset=cora,citeseer-clean,pubmed-testtype=nodes,edges-testprop=0.15,0.1:0.1:0.9-diml1enc=32-dimxiadj=16-bias=false-nepochs=4-nsamples=10.csv \
        --env PATH --env JULIA_NUM_THREADS --env OMP_NUM_THREADS \
        'cd ~/Code/nw2vec && python projects/behaviour/gae-benchmarks-graphsage.py --dataset datasets/gae-benchmarks/{dataset}.npz --testtype {tt} --testprop {tp} --diml1enc 32 --dimxiadj 16 --nepochs 4 --bias False --nworkers 1' \
        ::: sample $(seq 1 10) \
        ::: dataset cora citeseer-clean pubmed \
        ::: tt nodes edges \
        ::: tp 0.15 $(seq 0.1 0.1 0.9)
