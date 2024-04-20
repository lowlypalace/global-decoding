Example usage:

```
python global_decoding.py \
  --top_k 100 \
  --sequence_count 100 \
  --batch_size_seq 32 \
  --batch_size_prob 12 \
  --burnin 0.2 \
  --model gpt2 \
  --seed 0 \
  --output_dir output_$(date +%Y%m%d_%H%M) \
  --rate 1
```
