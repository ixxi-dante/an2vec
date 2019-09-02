#!/bin/bash -e

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base36-gpu
date
PATH=$PATH:~/bin OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 parallel \
        --sshloginfile parallel-slaves --sshdelay 0.1 \
        --bar --header : \
        --env PATH --env JULIA_NUM_THREADS --env OMP_NUM_THREADS \
        'cd ~/Code/nw2vec && OUTFOLDER=data/behaviour/gae-benchmarks/git=2b55e75a07-dataset=cora,citeseer-clean-testprop=0.15,0.1:0.1:0.9-testtype=nodes,edges-diml1enc=32-diml1dec=32-dimxiadj=32-dimxifeat=32-overlap=0,8,16,24,32-bias=false-sharedl1=false-decadjdeep=true,false-nepochs=200 && OUTPATH=$OUTFOLDER/dataset={dataset}-testtype={tt}-testprop={tp}-decadjdeep={decadjdeep}-dimxi={dimxi}-sample={sample}.npz && mkdir -p $OUTFOLDER && if test -e $OUTPATH; then echo "Already done"; false; fi && nice julia julia/an2vec-dataset.jl --dataset datasets/gae-benchmarks/{dataset}.npz --testtype {tt} --testprop {tp} --label-distribution {ld} --diml1enc 32 --diml1dec 32 --dimxiadj 32 --dimxifeat 32 --overlap $(( 64 - {dimxi} )) --nepochs 200 --bias false --sharedl1 false --decadjdeep {decadjdeep} --savehistory $OUTPATH' \
        ::: decadjdeep true false \
        ::: dataset cora citeseer-clean :::+ ld bernoulli bernoulli \
        ::: tt nodes edges \
        ::: tp 0.15 $(seq 0.1 0.1 0.9) \
        ::: dimxi $(seq 64 -8 32) \
        ::: sample $(seq 1 10)
