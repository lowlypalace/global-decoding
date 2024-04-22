Example usage to generate sequences:

```
python generate_sequences.py \
  --top_k 100 \
  --sequence_count 1000 \
  --batch_size_seq 128 \
  --max_length 10 \
  --batch_size_prob 16 \
  --model gpt2 \
  --seed 42
```

Example usage to run MCMC with the generated sequences:

```
python run_metropolis_hastings.py \
  --burnin 0.2 \
  --seed 0 \
  --dirs 22-04-2024_12-28-05 22-04-2024_12-36-21
```

Running tests:
```
python -m unittest
```
