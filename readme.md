Example usage:

```
python src/main.py \
  --top_k 100 \
  --sequence_count 100 \
  --batch_size_seq 8 \
  --batch_size_prob 8 \
  --model gpt2 \
  --seed 0 \
  --mcmc_rate 10 \
  --mcmc_burnin 0.2 \
  --eval_num_sequences 10 \
```

Running tests:
```
python -m unittest
```
