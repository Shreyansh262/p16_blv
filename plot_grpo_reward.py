import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# grpo/best has no trainer_state — use final checkpoint
with open("/usershome/cs671_user2/p16_blv/models/student/grpo/checkpoint-9076/trainer_state.json") as f:
    state = json.load(f)

log_history = state["log_history"]
print(f"Total log entries: {len(log_history)}")
print("Sample keys:", list(log_history[0].keys()) if log_history else "none")

steps, rewards, losses = [], [], []
reward_dir, reward_spatial, reward_hazard = [], [], []

for entry in log_history:
    step = entry.get("step", entry.get("global_step"))
    if step is None:
        continue
    steps.append(step)

    r = entry.get("reward") or entry.get("train/reward") or \
        entry.get("rewards") or entry.get("mean_reward")
    rewards.append(r)

    l = entry.get("loss") or entry.get("train/loss")
    losses.append(l)

    reward_dir.append(entry.get("reward_directional") or entry.get("directional_reward"))
    reward_spatial.append(entry.get("reward_spatial") or entry.get("spatial_reward"))
    reward_hazard.append(entry.get("reward_hazard") or entry.get("hazard_reward"))

print("\nAll unique keys in log_history:")
all_keys = set()
for e in log_history: all_keys.update(e.keys())
print(sorted(all_keys))

fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="#1a1a2e")
for ax in axes:
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

steps = np.array(steps)

def smooth(y, w=50):
    y = np.array([v if v is not None else np.nan for v in y], dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2: return y
    from scipy.ndimage import uniform_filter1d
    try:
        smoothed = uniform_filter1d(y[mask], size=min(w, mask.sum()//2 or 1))
        result = np.full_like(y, np.nan)
        result[mask] = smoothed
        return result
    except:
        return y

ax1 = axes[0]
has_reward = any(r is not None for r in rewards)
if has_reward:
    r_arr = np.array([r if r is not None else np.nan for r in rewards], dtype=float)
    ax1.plot(steps, r_arr, color="#ff6b35", alpha=0.3, linewidth=0.8, label="Raw reward")
    ax1.plot(steps, smooth(rewards), color="#ff6b35", linewidth=2, label="Smoothed reward")
    ax1.set_title("GRPO — Reward Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Reward", fontsize=11)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
else:
    ax1.text(0.5, 0.5, "Reward not logged separately\n(check key names printed above)",
             transform=ax1.transAxes, ha="center", va="center",
             color="#ff6b35", fontsize=12)
    ax1.set_title("GRPO — Reward (not found in logs)", fontsize=14)

ax2 = axes[1]
has_loss = any(l is not None for l in losses)
if has_loss:
    l_arr = np.array([l if l is not None else np.nan for l in losses], dtype=float)
    ax2.plot(steps, l_arr, color="#4cc9f0", alpha=0.3, linewidth=0.8, label="Raw loss")
    ax2.plot(steps, smooth(losses), color="#4cc9f0", linewidth=2, label="Smoothed loss")
    ax2.set_title("GRPO — Training Loss Curve", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")

ax2.set_xlabel("Training Step", fontsize=11)

plt.tight_layout(pad=2)
plt.savefig("/usershome/cs671_user2/p16_blv/grpo_reward_curve.png",
            dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print("\nSaved: ~/p16_blv/grpo_reward_curve.png")
