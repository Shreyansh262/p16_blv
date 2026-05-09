
import json, os
os.chdir('/usershome/cs671_user2/p16_blv')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

results = {}

for cond in ['A', 'B', 'C']:
    data = json.load(open(f'results/inference/conditions/condition_{cond}_outputs.json'))
    refs = {str(i): [x['reference']] for i, x in enumerate(data)}
    hyps = {str(i): [x['generated']] for i, x in enumerate(data)}
    scores = {}

    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(refs, hyps)
    scores['BLEU-1'] = round(bleu_score[0] * 100, 2)
    scores['BLEU-4'] = round(bleu_score[3] * 100, 2)

    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(refs, hyps)
    scores['METEOR'] = round(meteor_score * 100, 2)

    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(refs, hyps)
    scores['ROUGE-L'] = round(rouge_score * 100, 2)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    scores['CIDEr'] = round(cider_score, 4)

    results[cond] = scores
    print(f'Condition {cond} done.')

print()
print(f'{"Metric":<12} {"A (Base)":>12} {"B (SFT v2)":>12} {"C (DPO)":>12}')
print('-' * 52)
for metric in ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']:
    a = results['A'][metric]
    b = results['B'][metric]
    c = results['C'][metric]
    print(f'{metric:<12} {a:>12} {b:>12} {c:>12}')

json.dump(results, open('results/scores/nlp_metrics_ABC.json', 'w'), indent=2)
print()
print('Saved to results/scores/nlp_metrics_ABC.json')
