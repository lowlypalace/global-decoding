Example usage:

```
python global_decoding.py \
  --top_k 100 \
  --sequence_count 1000 \
  --batch_size_seq 128 \
  --max_length 10 \
  --batch_size_prob 16 \
  --burnin 0.2 \
  --model gpt2 \
  --seed 0 \
  --output_dir output \
  --rate 1
```

Running tests:
```
python -m unittest
```
