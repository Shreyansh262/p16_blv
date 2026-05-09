# results/

All evaluation outputs, scores, and analysis for the P16 BLV project.

FINAL EVALUATION: results/final_eval.png
(4-condition ablation: A=base, B=SFT_v2, C=SFT_v2+GRPO, D=sft_patch_v2)

---

## final_eval.png

Presentation-ready evaluation table comparing all 4 conditions across:
- NLP metrics (BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr)
- BLV keyword coverage (spatial, social, action, ambience, navigation)
- LLM judge scores (MCF, NAF and 8 sub-dimensions, scale 1-10)

Winner: D (sft_patch_v2) -- best on NLP, navigation, and overall LLM judge.

---

## inference/

Raw model-generated captions from evaluation runs.

    conditions/    condition_{A-E}_outputs.json -- full inference pipeline outputs
    balanced/      Balanced eval inference outputs (469 videos)
    comparisons/   Raw caption comparison runs (GRPO vs SFT variants)

---

## scores/

All scoring outputs -- LLM judge and NLP metrics.

    condition_{A-E}_llm_scores.json    Per-video LLM judge scores (6 dimensions)
    condition_{A-E}_llm_avg.json       Averaged scores per condition
    llm_judge_all.json                 Combined summary across all conditions
    nlp_metrics_ABC.json               BLEU/ROUGE/BERTScore for conditions A-C
    nlp_metrics_ABCD.json              BLEU/ROUGE/BERTScore for conditions A-D
    sft_patch_v2_*.json                SFT patch v2 dedicated eval outputs

---

## analysis/

Final result tables, reference datasets, and per-condition analysis.

    ground_truth.json           Ground-truth captions for held-out set
    held_out_test_set.json      100-video held-out test set (do not overwrite)
    final_results_table.json    Condensed final ablation results table
    final/                      Full per-video outputs and judge scores (all 4 models)
    new_conditions/             Additional condition inference outputs
    balanced_scores/            Balanced eval LLM judge scores per condition
    gt_scores/                  Ground-truth baseline scoring

---

## archive/

Superseded early-run outputs from April 5 (3-condition A-C run).
Kept for reference only.
