#!/bin/bash -e

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base36
date
#        --sshloginfile parallel-slaves --sshdelay 0.1 \
PATH=$PATH:~/bin OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 parallel \
        --bar --header : \
        -j50 \
        --env PATH --env JULIA_NUM_THREADS --env OMP_NUM_THREADS \
        'cd ~/Code/nw2vec && OUTFOLDER=data/behaviour/gae-benchmarks/git=15257d5eec-dataset=cora,citeseer-clean-testprop=0.15,0.1:0.1:0.9-testtype=nodes,edges-diml1enc=32-diml1dec=32-dimxiadj=8-dimxifeat=8-overlap=0,4,8-bias=false-sharedl1=false-decadjdeep=true,false-nepochs=200-nsamples=10 && OUTPATH=$OUTFOLDER/dataset={dataset}-testtype={tt}-testprop={tp}-decadjdeep={decadjdeep}-dimxi={dimxi}-sample={sample}.npz && mkdir -p $OUTFOLDER && if test -e $OUTPATH; then echo "Already done"; false; fi && julia julia/an2vec-dataset.jl --dataset datasets/gae-benchmarks/{dataset}.npz --testtype {tt} --testprop {tp} --label-distribution {ld} --diml1enc 32 --diml1dec 32 --dimxiadj 8 --dimxifeat 8 --overlap $(( 16 - {dimxi} )) --nepochs 200 --bias false --sharedl1 false --decadjdeep {decadjdeep} --savehistory $OUTPATH' \
        ::: sample $(seq 1 10) \
        ::: decadjdeep true false \
        ::: dataset cora citeseer-clean :::+ ld bernoulli bernoulli \
        ::: tt nodes edges \
        ::: tp 0.15 $(seq 0.1 0.1 0.9) \
        ::: dimxi $(seq 16 -4 8)
