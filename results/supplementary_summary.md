# Supplementary Experiment Summary

## 1. Question Type Breakdown

Seed 42 test split; methods: Flat Memory, Freq Promotion, Base SMC-QA, Retrieval-tuned SMC-QA.

### Base SMC-QA

| Question Type | Count | Hit@6 | Recall@6 | Contains |
|---------------|------:|------:|----------:|---------:|
| knowledge-update | 27 | 0.778 | 0.580 | 0.333 |
| multi-session | 36 | 0.750 | 0.485 | 0.222 |
| single-session-assistant | 17 | 0.588 | 0.588 | 0.529 |
| single-session-preference | 6 | 0.833 | 0.722 | 0.000 |
| single-session-user | 25 | 0.800 | 0.780 | 0.680 |
| temporal-reasoning | 39 | 0.769 | 0.544 | 0.154 |

### Retrieval-tuned SMC-QA

| Question Type | Count | Hit@6 | Recall@6 | Contains |
|---------------|------:|------:|----------:|---------:|
| knowledge-update | 27 | 0.741 | 0.543 | 0.333 |
| multi-session | 36 | 0.722 | 0.501 | 0.250 |
| single-session-assistant | 17 | 0.647 | 0.647 | 0.529 |
| single-session-preference | 6 | 0.833 | 0.722 | 0.000 |
| single-session-user | 25 | 0.800 | 0.780 | 0.680 |
| temporal-reasoning | 39 | 0.821 | 0.569 | 0.154 |

### Freq Promotion

| Question Type | Count | Hit@6 | Recall@6 | Contains |
|---------------|------:|------:|----------:|---------:|
| knowledge-update | 27 | 0.815 | 0.605 | 0.333 |
| multi-session | 36 | 0.806 | 0.512 | 0.222 |
| single-session-assistant | 17 | 0.588 | 0.588 | 0.529 |
| single-session-preference | 6 | 0.833 | 0.722 | 0.000 |
| single-session-user | 25 | 0.800 | 0.780 | 0.680 |
| temporal-reasoning | 39 | 0.821 | 0.599 | 0.103 |

## 2. LTM Quality

| Variant | LTM Size | Answer Item Rate | Gold Hit In LTM | Gold Recall In LTM | Redundancy |
|---------|---------:|-----------------:|----------------:|-------------------:|-----------:|
| Freq Promotion | 8.7 | 0.144 | 0.787 | 0.607 | 0.000 |
| Base SMC-QA | 13.6 | 0.100 | 0.800 | 0.650 | 0.000 |
| Retrieval-tuned SMC-QA | 10.1 | 0.120 | 0.780 | 0.603 | 0.000 |

## 3. Retrieval Depth Ablation

| Config | STM Top-K | LTM Top-K | Hit@6 | Recall@6 | Contains |
|--------|----------:|----------:|------:|----------:|---------:|
| baseline_thr0.40_top3_auto_stm6_ltm6 | 6 | 6 | 0.800 | 0.643 | 0.280 |
| baseline_thr0.40_top3_auto_stm8_ltm8 | 8 | 8 | 0.800 | 0.643 | 0.280 |
| div_heavy_thr0.40_top5_auto_stm6_ltm4 | 6 | 4 | 0.780 | 0.623 | 0.300 |
| baseline_thr0.25_top3_auto_stm6_ltm6 | 6 | 6 | 0.780 | 0.627 | 0.280 |
| baseline_thr0.25_top3_auto_stm8_ltm8 | 8 | 8 | 0.780 | 0.627 | 0.280 |

Merged test results for top retrieval-depth configs:

| Config | Hit@6 | Recall@6 | Contains |
|--------|------:|----------:|---------:|
| baseline_thr0.40_top3_auto_stm6_ltm6 | 0.789 +/- 0.033 | 0.628 +/- 0.024 | 0.329 +/- 0.033 |
| baseline_thr0.40_top3_auto_stm8_ltm8 | 0.793 +/- 0.030 | 0.631 +/- 0.024 | 0.329 +/- 0.033 |
| div_heavy_thr0.40_top5_auto_stm6_ltm4 | 0.769 +/- 0.019 | 0.606 +/- 0.017 | 0.327 +/- 0.033 |

## 4. Tuned Case Studies

Top cases where retrieval-tuned SMC-QA improves recall over Base SMC-QA:

- `single-session-assistant` I was going through our previous conversation about the impact of the political climate in Catalonia on its literature and music. Can you remind me of the example you gave of a Spanish-Catalan singer-songwriter who supports unity between Catalonia and Spain? (Base recall 0.00 -> Tuned recall 1.00)
- `multi-session` How many plants did I initially plant for tomatoes and cucumbers? (Base recall 0.50 -> Tuned recall 1.00)
- `temporal-reasoning` Which event did I participate in first, the charity gala or the charity bake sale? (Base recall 0.00 -> Tuned recall 0.50)

Top cases where retrieval-tuned SMC-QA regresses against Base SMC-QA:

- `knowledge-update` Where do I currently keep my old sneakers? (Base recall 1.00 -> Tuned recall 0.50)
- `knowledge-update` How many times have I met up with Alex from Germany? (Base recall 0.50 -> Tuned recall 0.00)
- `multi-session` How much did I spend on car wash and parking ticket? (Base recall 1.00 -> Tuned recall 0.50)
