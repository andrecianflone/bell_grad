#!/usr/bin/env bash

#set -x

envs=(twostateMDP GridWorld WindyGridWorld FrozenLake CliffWalking)
seeds=(0 1 2)
nepisodes=(2000)

for seed in ${seeds[@]}
do
    for env in ${envs[@]}
    do
        python main.py --pg_bellman --use_logger --seed ${seed[@]} --num_episodes ${nepisodes[@]} --env ${env[@]}
        python main.py --use_logger --seed ${seed[@]} --num_episodes ${nepisodes[@]} --env ${env[@]}
    done
done

