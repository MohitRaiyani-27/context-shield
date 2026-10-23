# context-shield benchmark (worst-case compliant model)

ASR = attack success rate (lower is better). Utility = benign queries whose correct document reaches the model.

| config | utility recall@5 ↑ | injection ASR ↓ | injection_heldout ASR ↓ | exfiltration ASR ↓ | poisoning ASR ↓ | flooding ASR ↓ | permission_probe ASR ↓ | citation_spoof ASR ↓ |
|---|---|---|---|---|---|---|---|---|
| naive | 100% | 100% | 100% | 100% | 100% | 100% | 94% | 100% |
| +acl | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 100% |
| +sanitize+detect | 100% | 0% | 100% | 0% | 100% | 100% | 0% | 100% |
| +ranking_guard | 100% | 0% | 50% | 0% | 0% | 0% | 0% | 100% |
| full (+spotlight+output_guard) | 100% | 0% | 50% | 0% | 0% | 0% | 0% | 0% |
| acl+output_guard only | 100% | 100% | 100% | 0% | 100% | 100% | 0% | 0% |

## Injection ASR by obfuscation style

| config | base64_wrapped | fake_delimiter | html_comment | multilingual_es | plain | polite_paraphrase | roleplay | unicode_tags | zero_width |
|---|---|---|---|---|---|---|---|---|---|
| naive | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| +acl | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| +sanitize+detect | 100% | 0% | 0% | 100% | 0% | 100% | 0% | 0% | 0% |
| +ranking_guard | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 0% |
| full (+spotlight+output_guard) | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 0% |
| acl+output_guard only | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

## Injection detector (standalone)

- Recall on dev attack documents (styles seen while writing the rules): **100%** (40 docs)
- Recall on held-out attack styles (never used for tuning): **0%** (18 docs)
- False positives on all benign passages: **2/82** (FPR 2.4%)
- False positives on hard negatives (legitimate text that sounds like an injection): **2/8**
