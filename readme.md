Example usage to generate sequences:

```
python generate_sequences.py \
  --top_k 100 \
  --sequence_count 100 \
  --batch_size_seq 4 \
  --batch_size_prob 4 \
  --model gpt2-large \
  --seed 0
```

Example usage to run MCMC with the generated sequences:

```
python run_metropolis_hastings.py \
  --burnin 0.2 \
  --seed 0 \
  --model gpt2-large \
  --top_k 100 \
  --rate 1
```

Running tests:
```
python -m unittest
```
