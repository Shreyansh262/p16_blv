import re

def compute_blv_reward(generated: str, reference: str) -> float:
    reward = 0.0
    gen_upper = generated.upper()
    ref_upper = reference.upper()

    ref_has_caution = "CAUTION" in ref_upper
    gen_has_caution = "CAUTION" in gen_upper
    if ref_has_caution and gen_has_caution:
        reward += 0.40
    elif ref_has_caution and not gen_has_caution:
        reward -= 0.30
    elif not ref_has_caution and gen_has_caution:
        reward -= 0.10

    gen_lower = generated.lower()
    direction_words = ["left", "right", "center", "ahead", "behind"]
    direction_count = sum(1 for w in direction_words if w in gen_lower)
    reward += min(direction_count * 0.08, 0.24)

    if re.search(r'\d+(\.\d+)?\s*(meter|metre)', gen_lower):
        reward += 0.20

    sentences = [s.strip() for s in generated.split('.') if s.strip()]
    if 2 <= len(sentences) <= 4:
        reward += 0.16
    elif len(sentences) > 6:
        reward -= 0.10

    hallucination_phrases = [
        "no visible text", "i cannot", "i can't", "no text present",
        "appears to be", "seems to be", "it seems", "i think",
        "going about day to day", "sharp without any"
    ]
    if any(p in gen_lower for p in hallucination_phrases):
        reward -= 0.20

    environment_words = [
        "kitchen", "living room", "bedroom", "bathroom", "hallway",
        "corridor", "office", "garage", "garden", "outdoor", "indoor",
        "staircase", "dining", "waiting area"
    ]
    first_sentence = generated.split('.')[0].lower() if generated else ""
    if any(w in first_sentence for w in environment_words):
        reward += 0.10

    return float(reward)


def unit_test():
    test_cases = [
        {
            "desc": "Perfect output — should score >= 0.8",
            "gen": "CAUTION: Kitchen environment, wet floor 1 meter ahead. Man in blue shirt moves left at 2 meters. Child stands center at 1.5 meters. Path right is clear.",
            "ref": "CAUTION: Kitchen environment. Wet floor at 1 meter. Man moves left.",
            "check": lambda s: s >= 0.8
        },
        {
            "desc": "Missing CAUTION — should score < 0.1",
            "gen": "Kitchen with a man and child. Man moves to the left side of the room.",
            "ref": "CAUTION: Kitchen environment. Wet floor at 1 meter.",
            "check": lambda s: s < 0.1
        },
        {
            "desc": "Hallucination output — should score <= -0.3",
            "gen": "A baby lies on its back. It seems relaxed. No visible text present in the image.",
            "ref": "CAUTION: Living room. Toys on floor 1 meter ahead.",
            "check": lambda s: s <= -0.3
        },
        {
            "desc": "Good output no CAUTION needed — should score >= 0.5",
            "gen": "Living room. Person in white shirt seated 2 meters ahead center. Coffee table at 1 meter right. Path left is clear.",
            "ref": "Living room. Person in white shirt 2 meters ahead.",
            "check": lambda s: s >= 0.5
        },
    ]
    print("=== Reward Unit Tests ===")
    all_passed = True
    for tc in test_cases:
        score = compute_blv_reward(tc["gen"], tc["ref"])
        passed = tc["check"](score)
        if not passed:
            all_passed = False
        print(f"{'PASS' if passed else 'FAIL'} | Score: {score:.3f} | {tc['desc']}")
    print("All passed" if all_passed else "SOME FAILED — stop here")
    return all_passed

if __name__ == "__main__":
    unit_test()
