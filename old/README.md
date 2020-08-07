BGrad

Launch standard pg actor critic
```
python main.py --use_logger --seed 0 --num_episodes 100 --env GridWorld
```

Launch bellman pg actor critic
```
python main.py --pg_bellman --use_logger --seed 0 --num_episodes 100 --env GridWorld
```

Run batch unordered parallel jobs over all environments and seeds
```
./test.sh
```
