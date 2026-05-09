
import json, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/usershome/cs671_user2/p16_blv')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import nltk
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.tokenize import word_tokenize

# Load existing A/B/C results
existing = json.load(open('results/scores/nlp_metrics_ABC.json'))
results = dict(existing)

# Compute D only
for cond in ['D']:
    data = json.load(open(f'results/inference/conditions/condition_{cond}_outputs.json'))
    refs = {str(i): [x['reference']] for i, x in enumerate(data)}
    hyps = {str(i): [x['generated']] for i, x in enumerate(data)}
    scores = {}

    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(refs, hyps)
    scores['BLEU-1'] = round(bleu_score[0] * 100, 2)
    scores['BLEU-4'] = round(bleu_score[3] * 100, 2)

    meteor_vals = []
    for i in range(len(data)):
        ref_tokens = [word_tokenize(refs[str(i)][0].lower())]
        hyp_tokens = word_tokenize(hyps[str(i)][0].lower())
        meteor_vals.append(nltk_meteor(ref_tokens, hyp_tokens))
    scores['METEOR'] = round(sum(meteor_vals) / len(meteor_vals) * 100, 2)

    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(refs, hyps)
    scores['ROUGE-L'] = round(rouge_score * 100, 2)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    scores['CIDEr'] = round(cider_score, 4)

    results[cond] = scores
    print(f'Condition {cond} done.')

# Print full 4-condition table
labels = {
    'A': 'A (Base)',
    'B': 'B (SFT v2)',
    'C': 'C (DPO)',
    'D': 'D (RLAIF-DPO)',
}
print()
print(f'{"Metric":<12} {"A (Base)":>12} {"B (SFT v2)":>14} {"C (DPO)":>10} {"D (RLAIF)":>12}')
print('-' * 64)
for metric in ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']:
    vals = [results[c][metric] for c in ['A','B','C','D']]
    best = max(vals)
    def fmt(v): return f'*{v}*' if v == best else str(v)
    print(f'{metric:<12} {fmt(vals[0]):>12} {fmt(vals[1]):>14} {fmt(vals[2]):>10} {fmt(vals[3]):>12}')

json.dump(results, open('results/scores/nlp_metrics_ABCD.json', 'w'), indent=2)
print()
print('Saved to results/scores/nlp_metrics_ABCD.json')
