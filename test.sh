#!/usr/bin/env bash
# The code here runs the jobs in parallel, unordered. For FIFO ordered, see:
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop


# Settings
mkdir -p logs
i=0
count=0
now=logs/$(date +"%Y%m%d%H%M")
p_jobs=20 # max parallel jobs

# Job settings
envs=(twostateMDP GridWorld WindyGridWorld FrozenLake CliffWalking)
#envs=(twostateMDP)
seeds=(0 1 2 3 4 5 6 7 8 9)
nepisodes=(3000)

(
for seed in ${seeds[@]}
do
    for env in ${envs[@]}
    do
        ((count++))
        # PG Bellman
        ((i=i%p_jobs)); ((i++==0)) && wait # job lock
        echo "forking"

        # GradNet
        if [ -n "$1" ]; then
            # Debugging
            if [[ $1 == *"debug"* ]]; then
            echo "debugging"
            python -m pudb main.py \
                --pg_bellman --seed ${seed[@]} \
                --num_episodes ${nepisodes[@]} --env ${env[@]}
            exit 0
            fi
        else
            # No debugging
            python main.py \
                --pg_bellman --seed ${seed[@]} --use_logger \
                --num_episodes ${nepisodes[@]} --env ${env[@]} >"${now}_ori${count}.txt" &
        fi

        # Run a baseline to compare with
        ((i=i%p_jobs)); ((i++==0)) && wait # job lock
        echo "forking"
        if [ -n "$1" ]; then
            # Debugging
            if [[ $1 == *"debug"* ]]; then
            echo "debugging"
            python -m pudb main.py \
                --seed ${seed[@]} --use_logger \
                --num_episodes ${nepisodes[@]} --env ${env[@]}
            fi
        else
            python main.py \
                --seed ${seed[@]} --use_logger \
                --num_episodes ${nepisodes[@]} --env ${env[@]} >"${now}_pgb${count}.txt" &
        fi

        # Wait a sec otherwise race condition in directory creation
        sleep 0.5
    done
done
)
