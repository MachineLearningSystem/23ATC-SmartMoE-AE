#!/bin/bash

export AEROOT=`pwd`
./clean.sh
output_dir=$AEROOT/RUNME-b-output_$(date -Iseconds)
mkdir $output_dir

cd ./plotting/from_exec/fig8
./fig8.sh
cd -
AEROOT=`pwd` python3 ./plotting/from_exec/fig8.py `pwd` $output_dir

cd ./plotting/from_exec/fig10
./fig10.sh
cd -
AEROOT=`pwd` python3 ./plotting/from_exec/fig10.py `pwd` $output_dir

cd ./plotting/from_exec/fig12
./fig12.sh
cd -
AEROOT=`pwd` python3 ./plotting/from_exec/fig12.py `pwd` $output_dir

cd ./plotting/from_exec/fig13
./fig13.sh
cd -
AEROOT=`pwd` python3 ./plotting/from_exec/fig13.py `pwd` $output_dir