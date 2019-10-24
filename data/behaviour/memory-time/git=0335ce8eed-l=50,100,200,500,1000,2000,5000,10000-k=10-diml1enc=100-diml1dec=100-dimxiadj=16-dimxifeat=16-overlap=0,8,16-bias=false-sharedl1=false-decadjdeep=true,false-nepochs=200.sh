#!/bin/bash -e

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate base36
date
#        --sshloginfile parallel-slaves --sshdelay 0.1 \
PATH=$PATH:~/bin OMP_NUM_THREADS=8 JULIA_NUM_THREADS=8 parallel \
        --bar --header : \
        --results "/home/slerique/Code/nw2vec/data/behaviour/memory-time/git=0335ce8eed-l=50,100,200,500,1000,2000,5000,10000-k=10-diml1enc=100-diml1dec=100-dimxiadj=16-dimxifeat=16-overlap=0,8,16-bias=false-sharedl1=false-decadjdeep=true,false-nepochs=200.csv" \
        -j1 \
        --env PATH --env JULIA_NUM_THREADS --env OMP_NUM_THREADS \
        'cd ~/Code/nw2vec && julia julia/an2vec-generative.jl -l {l} -k {k} --p_in 0.25 --p_out 0.01 --correlation 0.7 --featuretype colors --diml1enc 100 --diml1dec 100 --dimxiadj 16 --dimxifeat 16 --overlap $(( 32 - {dimxi} )) --nepochs 200 --bias false --sharedl1 false --decadjdeep {decadjdeep} --savehistory /dev/null' \
        ::: decadjdeep true false \
        ::: l 50 100 200 500 1000 2000 5000 10000 \
        ::: k 10 \
        ::: dimxi $(seq 32 -8 16)
