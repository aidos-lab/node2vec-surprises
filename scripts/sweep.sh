#!/usr/bin/env bash
#
# Runs a 'sweep' over different parameter configurations. Results are
# stored automatically.

NAME="N2VS"

for CONTEXT in 5 10; do
  for DIMENSION in 16 32 64; do
    for LENGTH in 5 10 20; do
      for PROBABILITY in 0.30 0.50; do
        sbatch -p cpu_p               \
               -J ${NAME}             \
               -o "${NAME}_%j.out"    \
               --cpus-per-task=4      \
               --mem=4G               \
               --nice=10000           \
               --wrap "poetry run python ../rawos/node2vec.py --length $LENGTH --dimension $DIMENSION --context $CONTEXT --edge-prob $PROBABILITY --epochs 200 --keep "
      done
    done
  done
done
