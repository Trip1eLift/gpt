# gpt
Implementing my own GPT model

Parameters: d_m x d_in + 2 x d_m x seq + N (4 x d_m^2 + 4 x d_m x seq + 2 x d_m x d_h) + d_m x d_in

d_h = 2 d_m, d_hidden in feedforward

## shakeGPTv1

| d_model | d_input | layers | num_heads | dropout | Batch size | max_seq_len | learning rate | total iters | val loss | position function |
|---------|---------|--------|-----------|---------|------------|-------------|---------------|-------------|----------| ----------------- |
|      32 |      66 |     12 |         8 |     0.1 |         32 |         200 |       0.00003 |      253104 |   1.7002 |       PE function |

Total parameters = 32x66+2x32x200+12(4x32^2+4x32x200+4x32^2)+32x66 = 422,528

```text
T+ 5:20:32 - step 253104/453103: train loss 1.5575, val loss 1.7002; train acc  0.4090, val acc 0.3735
```

## shakeGPTv2

| d_model | d_input | layers | num_heads | dropout | Batch size | max_seq_len | learning rate | total iters | val loss | position function |
|---------|---------|--------|-----------|---------|------------|-------------|---------------|-------------|----------| ----------------- |
|     384 |      66 |      6 |         6 |     0.2 |         64 |         256 |       0.00003 |        7270 |   1.7531 |       PE function |

Total parameters = 384x66+2x384x256+6(4x384^2+4x384x256+4x384^2)+384x66 = 9,684,480

```text
T+ 0:37:46 - step 7270/7270: train loss 1.5866, val loss 1.7531; train acc  0.3996, val acc 0.3625; t_remain 0:0:00
```

## shakeGPTv3

| d_model | d_input | layers | num_heads | dropout | Batch size | max_seq_len | learning rate | total iters | val loss | position function |
|---------|---------|--------|-----------|---------|------------|-------------|---------------|-------------|----------| ----------------- |
|     384 |      66 |      6 |         6 |     0.2 |         64 |         256 |       0.00003 |        7657 |   1.6900 |         Embedding |

Total parameters = 384x66+2x384x256+6(4x384^2+4x384x256+4x384^2)+384x66+256x384 = 9,782,784

```text
T+ 0:40:01 - step 7657/9652: train loss 1.5231, val loss 1.6900; train acc  0.4141, val acc 0.3893; t_remain 0:10:41
```
