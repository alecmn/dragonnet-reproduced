#!/usr/bin/env bash

# script should be run from 'src'

list=(
    dragonnet
    tarnet

)

folders=(
    a
)

for i in ${list[@]}; do
    for folder in ${folders[@]}; do
        echo $i
        echo $folder
        python -m experiment.acic_main2 --data_base_dir ../dat/LIBDD\
                                     --knob $i\
                                     --folder $folder\
                                     --output_base_dir ../result/acic\

    done

done
