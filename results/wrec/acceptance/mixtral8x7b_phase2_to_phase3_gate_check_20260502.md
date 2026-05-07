# Mixtral Phase Gate Check

## Verdict

The current Mixtral WREC trace results satisfy the planned gate for continuing beyond trace collection.

## Gate Results

| Gate | Result | Evidence |
|---|---:|---|
| Mixtral offload probe succeeded | pass | `48GiB n=4` probe completed with `4` requests, `6048` router events, `0` failures. |
| Eval trace valid requests >= 240 | pass | Eval trace completed `256` requests. |
| Trace failure rate <= 5% | pass | Eval trace failures: `0/256 = 0%`. |
| >= 80% MoE layers have non-uniform hotness | pass | Train: `32/32`; eval: `32/32` layers pass chi-square against uniform routing. |
| Short-window locality beats random baseline | pass | Eval mean unique experts per layer window is below random for windows `4/8/16`. |

## Hotness Check

Non-uniform hotness uses chi-square against uniform expert frequency over `8` experts per layer, with `df=7` and threshold `chi-square > 24.3`.

| trace | passing layers | chi-square min | chi-square p50 | chi-square max | top1 share p50 |
|---|---:|---:|---:|---:|---:|
| train | 32/32 | 177.6550 | 1866.0016 | 6452.9849 | 0.1611 |
| eval | 32/32 | 47.3371 | 1018.8279 | 2932.4124 | 0.1641 |

## Short-Window Locality

Random baseline assumes each router event independently selects `2` experts uniformly without replacement from `8` experts. Lower unique expert count means stronger locality.

| trace | window | mean unique experts | random expected | improvement |
|---|---:|---:|---:|---:|
| train | 4 | 4.9503 | 5.4688 | 9.48% |
| train | 8 | 6.5806 | 7.1991 | 8.59% |
| train | 16 | 7.5851 | 7.9198 | 4.23% |
| eval | 4 | 4.9423 | 5.4688 | 9.63% |
| eval | 8 | 6.5724 | 7.1991 | 8.71% |
| eval | 16 | 7.5831 | 7.9198 | 4.25% |

## Conclusion

The trace gate is satisfied. The next step can proceed to WREC-H implementation and evaluation against `lru`, `static_hot`, route-window variants, and `belady_oracle`.
