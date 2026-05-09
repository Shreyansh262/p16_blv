import os, json, logging, torch
import torch.nn.functional as F
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, SmolVLMForConditionalGeneration, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/dpo_training.log"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("=== DPO TRAINING START === " + datetime.now().isoformat())

if torch.cuda.is_available():
    logger.info("GPU: " + torch.cuda.get_device_name(0))
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free  = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
    logger.info(f"VRAM total: {total:.1f} GB | free: {free:.1f} GB")

SFT_CHECKPOINT = "models/student/sft_v2/best/"
BASE_MODEL     = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OUTPUT_DIR     = "models/student/dpo_v2_sft_v2"
PROMPT_TEXT    = (
    "You are an audio description assistant for blind and low-vision users. "
    "Describe this scene for safe navigation: mention ALL floor-level hazards and trip risks FIRST, "
    "then spatial layout with distances, then people positions and directions of movement. "
    "Use CAUTION prefix for any hazard. Be concise."
)

# 364 = 14*26: controls max tile count. 4 images at 364px → ~52 tiles → ~3328 image tokens.
# MAX_SEQ=512 (old) truncated through image tokens → divisibility check failed every batch.
# MAX_SEQ=4096 fits the full 3478-token prompt + 128 response tokens comfortably.
MAX_IMAGE_SIDE = 364
BETA       = 0.3
GRAD_ACCUM = 8
LR         = 1e-7
WARMUP     = 50
EVAL_STEPS = 200
SAVE_STEPS = 200
LOG_STEPS  = 10
MAX_SEQ    = 4096
MAX_RESP   = 128

logger.info("Loading processor from SFT checkpoint...")
processor = AutoProcessor.from_pretrained(SFT_CHECKPOINT)

logger.info("Loading and formatting DPO pairs...")
dpo_data = json.load(open("data/generated/dpo_pairs_v2.json"))


def process_split(entries, split_name):
    records, skipped = [], 0
    for i, entry in enumerate(entries):
        if i % 500 == 0:
            logger.info(f"  {split_name}: processing {i}/{len(entries)}...")
        kfps = entry.get("keyframe_paths", [])
        if not kfps or any(not os.path.exists(p) for p in kfps):
            skipped += 1
            continue
        records.append({
            "chosen":         entry["chosen"],
            "rejected":       entry["rejected"],
            "keyframe_paths": kfps,
        })
    logger.info(f"  {split_name}: {len(records)} kept, {skipped} skipped")
    return records


train_records = process_split(dpo_data["train"], "train")
val_records   = process_split(dpo_data["val"],   "val")
train_dataset = Dataset.from_list(train_records)
eval_dataset  = Dataset.from_list(val_records)
logger.info(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")


def load_images(keyframe_paths):
    images = []
    for p in keyframe_paths:
        img = Image.open(p).convert("RGB")
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
        images.append(img)
    return images


class MultimodalDPOCollator:
    """apply_chat_template(PIL images) + processor() with same images.
    The processor inserts N image tokens matching the tiles it will create,
    so input_ids image count == num_tiles * tokens_per_tile (divisibility passes).
    MAX_SEQ must be large enough to hold the full prompt without truncating image tokens.
    """

    def __init__(self, processor, prompt_text, max_seq=MAX_SEQ, max_resp=MAX_RESP):
        self.processor   = processor
        self.prompt_text = prompt_text
        self.max_seq     = max_seq
        self.max_resp    = max_resp

    def __call__(self, batch):
        b = batch[0]  # batch_size == 1

        images = load_images(b["keyframe_paths"])

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
                       + [{"type": "text", "text": self.prompt_text}],
        }]
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_enc = self.processor(
            text=[prompt_str], images=images, return_tensors="pt",
        )
        prompt_ids  = prompt_enc["input_ids"][0]
        prompt_mask = prompt_enc["attention_mask"][0]
        pv          = prompt_enc.get("pixel_values")
        pa          = prompt_enc.get("pixel_attention_mask")

        # Guard: truncating image tokens breaks the model divisibility check.
        # Skip any sample whose prompt alone cannot fit in MAX_SEQ.
        if len(prompt_ids) >= self.max_seq:
            logger.warning(f"Skipping: prompt length {len(prompt_ids)} >= MAX_SEQ {self.max_seq}")
            return {"_skip": True}

        out = {}
        if pv is not None: out["pixel_values"]         = pv
        if pa is not None: out["pixel_attention_mask"] = pa

        for key in ("chosen", "rejected"):
            resp_ids = self.processor.tokenizer(
                b[key], return_tensors="pt",
                truncation=True, max_length=self.max_resp,
                add_special_tokens=False,
            )["input_ids"][0]

            full_ids  = torch.cat([prompt_ids, resp_ids])[:self.max_seq]
            full_mask = torch.cat([prompt_mask, torch.ones_like(resp_ids)])[:self.max_seq]
            resp_mask = torch.cat([
                torch.zeros(len(prompt_ids), dtype=torch.long),
                torch.ones( len(resp_ids),   dtype=torch.long),
            ])[:self.max_seq]

            out[f"{key}_input_ids"]      = full_ids.unsqueeze(0)
            out[f"{key}_attention_mask"] = full_mask.unsqueeze(0)
            out[f"{key}_response_mask"]  = resp_mask.unsqueeze(0)

        return out


def get_sequence_logps(model, input_ids, attention_mask, response_mask,
                       pixel_values=None, pixel_attention_mask=None):
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if pixel_values         is not None: kwargs["pixel_values"]          = pixel_values
    if pixel_attention_mask is not None: kwargs["pixel_attention_mask"]  = pixel_attention_mask
    logits       = model(**kwargs).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_rmask  = response_mask[:, 1:]
    token_logps  = F.log_softmax(shift_logits, dim=-1).gather(
                       -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shift_rmask).sum(-1)


def _pixel_kwargs(batch, device):
    kw = {}
    pv = batch.get("pixel_values")
    pa = batch.get("pixel_attention_mask")
    if pv is not None: kw["pixel_values"]         = pv.to(device=device, dtype=torch.bfloat16)
    if pa is not None: kw["pixel_attention_mask"] = pa.to(device)
    return kw


def _seq_kwargs(batch, key, device):
    return dict(
        input_ids      = batch[f"{key}_input_ids"].to(device),
        attention_mask = batch[f"{key}_attention_mask"].to(device),
        response_mask  = batch[f"{key}_response_mask"].to(device),
    )


def run_eval(model, model_ref, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch.get("_skip"):
                continue
            try:
                kw = _pixel_kwargs(batch, device)
                c  = _seq_kwargs(batch, "chosen",  device)
                r  = _seq_kwargs(batch, "rejected", device)
                pc = get_sequence_logps(model,     **c, **kw)
                pr = get_sequence_logps(model,     **r, **kw)
                rc = get_sequence_logps(model_ref, **c, **kw)
                rr = get_sequence_logps(model_ref, **r, **kw)
                loss = -F.logsigmoid(BETA * ((pc - rc) - (pr - rr))).mean()
                total += loss.item(); n += 1
            except Exception:
                pass
    model.train()
    return total / max(n, 1)


def load_sft_model():
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    return PeftModel.from_pretrained(base, SFT_CHECKPOINT).merge_and_unload()


logger.info("Loading policy model (SFT merged)...")
model = load_sft_model()

logger.info("Loading reference model (SFT merged, frozen)...")
model_ref = load_sft_model()
model_ref.eval()
for p in model_ref.parameters():
    p.requires_grad = False

lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

DEVICE = next(model.parameters()).device
os.makedirs(OUTPUT_DIR, exist_ok=True)

collator     = MultimodalDPOCollator(processor, PROMPT_TEXT)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  collate_fn=collator, num_workers=0)
eval_loader  = DataLoader(eval_dataset,  batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

n_grad_steps = len(train_loader) // GRAD_ACCUM
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP,
                                            num_training_steps=n_grad_steps)

logger.info(f"Train batches: {len(train_loader)} | Grad steps: {n_grad_steps}")
logger.info("=== STARTING DPO TRAINING ===")

model.train()
optimizer.zero_grad()
global_step  = 0
running_loss = 0.0
best_eval    = float("inf")

for i, batch in enumerate(train_loader):
    if batch.get("_skip"):
        continue
    try:
        kw = _pixel_kwargs(batch, DEVICE)
        c  = _seq_kwargs(batch, "chosen",  DEVICE)
        r  = _seq_kwargs(batch, "rejected", DEVICE)

        pol_c = get_sequence_logps(model, **c, **kw)
        pol_r = get_sequence_logps(model, **r, **kw)
        with torch.no_grad():
            ref_c = get_sequence_logps(model_ref, **c, **kw)
            ref_r = get_sequence_logps(model_ref, **r, **kw)

        loss = -F.logsigmoid(BETA * ((pol_c - ref_c) - (pol_r - ref_r))).mean() / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM

    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM at batch {i} — clearing cache and skipping")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        continue
    except Exception as ex:
        logger.warning(f"Skipping batch {i}: {ex}")
        optimizer.zero_grad()
        continue

    if (i + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if global_step % LOG_STEPS == 0:
            avg = running_loss / LOG_STEPS
            logger.info(f"step={global_step}/{n_grad_steps}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0

        if global_step % EVAL_STEPS == 0:
            eval_loss = run_eval(model, model_ref, eval_loader, DEVICE)
            logger.info(f"step={global_step}  eval_loss={eval_loss:.4f}  best={best_eval:.4f}")
            if eval_loss < best_eval:
                best_eval = eval_loss
                model.save_pretrained(f"{OUTPUT_DIR}/best")
                logger.info(f"  -> new best saved")

        if global_step % SAVE_STEPS == 0:
            ckpt = f"{OUTPUT_DIR}/checkpoint-{global_step}"
            model.save_pretrained(ckpt)
            logger.info(f"Checkpoint saved: {ckpt}")

model.save_pretrained(f"{OUTPUT_DIR}/best")
logger.info(f"Final model saved. Best eval loss: {best_eval:.4f}")
logger.info("=== DPO TRAINING COMPLETE === " + datetime.now().isoformat())
