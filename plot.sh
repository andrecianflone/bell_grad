python plot.py \
    --paths "./results/pg_bellman/GridWorld/*/returns_eval.npy" \
            "./results/baseline/GridWorld/*/returns_eval.npy" \
    --labels "Bellman PG Update" "Original PG Update" \
    --xlabel "Episodes" \
    --ylabel "Cumulative Returns" \
    --title "GridWorld, Actor-Critic (Non-Linear)"

