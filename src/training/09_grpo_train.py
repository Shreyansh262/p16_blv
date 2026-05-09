import json, os, sys, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

SFT_CHECKPOINT = "models/student/sft_v2/best"
TRAIN_DATA     = "data/generated/grpo_train.json"
VAL_DATA       = "data/generated/grpo_val.json"
OUTPUT_DIR     = "models/student/grpo"
BASE_MODEL_ID  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

sys.path.insert(0, os.getcwd())
from src.training.grpo_reward import compute_blv_reward


class BLVDataset(Dataset):
    def __init__(self, data_path, processor, max_samples=None):
        with open(data_path) as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image  = Image.open(sample["keyframe_path"]).convert("RGB")
        # TRL 0.29.1 requires raw conversational format â€” it applies chat template internally
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["prompt"]}
            ]
        }]
        return {
            "prompt":          messages,
            "images":          [image],
            "gemma_reference": sample["gemma_reference"],
            "sample_id":       sample["sample_id"],
        }


def make_reward_fn(dataset):
    def reward_fn(completions, prompts=None, gemma_reference=None, **kwargs):
        rewards = []
        refs = gemma_reference if gemma_reference is not None else [""] * len(completions)
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            reference = refs[i] if i < len(refs) else ""
            rewards.append(compute_blv_reward(text, reference))
        return rewards
    return reward_fn


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    print(f"Loading SFT checkpoint: {SFT_CHECKPOINT}")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        model,
        SFT_CHECKPOINT,
        is_trainable=True
    )
    model.enable_adapter_layers()
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset = BLVDataset(TRAIN_DATA, processor)
    val_dataset   = BLVDataset(VAL_DATA,   processor, max_samples=100)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4,
        max_completion_length=256,
        learning_rate=5e-6,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        weight_decay=0.01,
        beta=0.1,
        logging_steps=10,
        save_steps=250,
        eval_steps=250,
        save_total_limit=3,
        seed=42,
        bf16=True,
    )

    reward_fn = make_reward_fn(train_dataset)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )

    print("=== Starting GRPO Training ===")
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/best")
    print(f"Done. Saved to {OUTPUT_DIR}/best")

if __name__ == "__main__":
    main()
