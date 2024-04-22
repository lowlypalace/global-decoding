Example usage to generate sequences:

```
python generate_sequences.py \
  --top_k 100 \
  --sequence_count 1000 \
  --batch_size_seq 128 \
  --max_length 10 \
  --batch_size_prob 16 \
  --model gpt2 \
  --seed 0 \
```

Example usage to run Metropolis-Hastings with the generated sequences:

```
python run_metropolis_hastings.py \
  --burnin 0.2 \
  --seed 0 \
  --dirs 22-04-2024_11-43-31
```

Running tests:
```
python -m unittest
```
