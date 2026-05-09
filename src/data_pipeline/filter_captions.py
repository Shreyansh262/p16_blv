import re
import json
from difflib import SequenceMatcher
from itertools import combinations

# ── CONFIG ──────────────────────────────────────────────────────────────
INPUT_JSON  = "data/generated/all_captions_gemma.json"   # your Gemma output file
OUTPUT_GOOD = "data/generated/captions_filtered.json"
OUTPUT_BAD  = "data/generated/captions_rejected.json"
MIN_SCORE   = 40   # drop captions below this BLV score
# ────────────────────────────────────────────────────────────────────────

HALLUCINATION_PHRASES = [
    'warm atmosphere', 'communal living', 'familial', 'exudes',
    'socialization', 'suggests ', 'appears to be', 'seemingly',
    'reminiscent of', 'implies', 'evokes', 'sense of', 'feeling of',
    'mood of', 'conveys', 'symbolizes', 'metaphor', 'atmosphere'
]

AD_VIOLATIONS = {
    'past_tense':     (r'\b(was|were|had been|looked like|seemed to)\b',
                       "Past tense used (AD must be present tense)"),
    'emotion_attr':   (r'\b(seemed happy|appeared (angry|sad|scared|excited)|felt |wanted to)\b',
                       "Emotion/intent attributed to subject"),
    'camera_lang':    (r'\b(we see|camera (shows|pans|zooms)|visible in frame|in the (video|clip|scene|frame))\b',
                       "Camera/viewer language used"),
    'mind_reading':   (r'\b(thinking|wondering|decided to|trying to|about to|planning to)\b',
                       "Mind-reading language used"),
}

def blv_score(caption):
    score = 0
    text = caption.lower()
    rooms = ['kitchen','bedroom','bathroom','living room',
             'hallway','corridor','office','outdoor','street','garden','staircase']
    if any(r in text for r in rooms): score += 20
    spatial = ['left','right','ahead','behind','center','near','far','beside','above','below']
    score += min(20, sum(1 for s in spatial if s in text) * 5)
    if re.search(r'\d+\s*(meter|feet|foot|cm|step)', text): score += 20
    elif any(w in text for w in ['height','waist','knee','shoulder']): score += 10
    hazards = ['hazard','caution','careful','obstacle','trip','wet','step','slippery','low','sharp']
    if any(h in text for h in hazards): score += 20
    sentences = text.count('.')
    words = len(text.split())
    if 2 <= sentences <= 4 and words <= 80: score += 20
    return score

def check_hallucination(caption):
    text = caption.lower()
    hits = [p for p in HALLUCINATION_PHRASES if p in text]
    return hits  # empty list = clean

def check_repetition(caption, threshold=0.75):
    sentences = [s.strip() for s in caption.split('.') if len(s.strip()) > 10]
    for a, b in combinations(sentences, 2):
        if SequenceMatcher(None, a, b).ratio() > threshold:
            return True
    return False

def check_ad_guidelines(caption):
    violations = []
    for rule_name, (pattern, message) in AD_VIOLATIONS.items():
        if re.search(pattern, caption.lower()):
            violations.append(message)
    return violations  # empty list = compliant

def check_length(caption):
    words = len(caption.split())
    if words < 15:  return "Too short (< 15 words)"
    if words > 300: return "Too long (> 300 words)"
    return None

# ── MAIN ────────────────────────────────────────────────────────────────
with open(INPUT_JSON) as f:
    data = json.load(f)  # expects list of {"video_id": ..., "caption": ...}

good, bad = [], []

for item in data:
    vid   = item["video_id"]
    cap   = item["blv_description"]
    flags = []

    # 1. Length check
    length_issue = check_length(cap)
    if length_issue:
        flags.append(length_issue)

    # 2. BLV score
    score = blv_score(cap)
    if score < MIN_SCORE:
        flags.append(f"Low BLV score: {score}/100")

    # 3. Hallucination phrases
    hits = check_hallucination(cap)
    if hits:
        flags.append(f"Hallucination phrases: {hits}")

    # 4. Repetition
    if check_repetition(cap):
        flags.append("Repetitive sentences detected")

    # 5. AD Guidelines
    ad_violations = check_ad_guidelines(cap)
    for v in ad_violations:
        flags.append(f"AD violation: {v}")

    result = {**item, "blv_score": score, "flags": flags}
    (bad if flags else good).append(result)

with open(OUTPUT_GOOD, 'w') as f: json.dump(good, f, indent=2)
with open(OUTPUT_BAD,  'w') as f: json.dump(bad,  f, indent=2)

# ── REPORT ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Total captions   : {len(data)}")
print(f"✅ Kept (clean)  : {len(good)} ({100*len(good)/len(data):.1f}%)")
print(f"❌ Rejected      : {len(bad)}  ({100*len(bad)/len(data):.1f}%)")

from collections import Counter
all_flags = [f for item in bad for f in item['flags']]
print("\nTop rejection reasons:")
for reason, count in Counter(all_flags).most_common(10):
    print(f"  {count:>5}x  {reason}")
