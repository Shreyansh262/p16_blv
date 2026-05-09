import os, logging, torch
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
from datasets import load_dataset
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
        logging.FileHandler("logs/rlaif_dpo.log"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("=== RLAIF-V DPO START === " + datetime.now().isoformat())

SFT_CHECKPOINT = "models/student/dpo_v2_sft_v2/checkpoint-800"
BASE_MODEL      = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OUTPUT_DIR      = "models/student/rlaif_dpo"
BETA            = 0.3
LR              = 1e-7
GRAD_ACCUM      = 8
MAX_SEQ         = 4096
MAX_RESP        = 256
LOG_STEPS       = 10
EVAL_STEPS      = 200
SAVE_STEPS      = 200
NUM_SAMPLES     = 10000   # subset of 83K — enough signal, manageable time

logger.info("Loading processor...")
# checkpoint-800 only has adapter weights; load processor from base model
try:
    processor = AutoProcessor.from_pretrained(SFT_CHECKPOINT)
except Exception:
    logger.info("Processor not in checkpoint, loading from base model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

# Load RLAIF-V subset
logger.info(f"Loading RLAIF-V dataset ({NUM_SAMPLES} samples)...")
raw = load_dataset("openbmb/RLAIF-V-Dataset", split="train", streaming=False)
raw = raw.shuffle(seed=42).select(range(NUM_SAMPLES))
split = raw.train_test_split(test_size=0.05, seed=42)
train_data = split["train"]
val_data   = split["test"]
logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")


class RLAIFCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        b = batch[0]
        image = b["image"].convert("RGB")
        question = b["question"]
        chosen   = b["chosen"]
        rejected = b["rejected"]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question}
        ]}]
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_enc = self.processor(
            text=[prompt_str], images=[image], return_tensors="pt"
        )
        prompt_ids  = prompt_enc["input_ids"][0]
        prompt_mask = prompt_enc["attention_mask"][0]
        pv = prompt_enc.get("pixel_values")
        pa = prompt_enc.get("pixel_attention_mask")

        out = {}
        if pv is not None: out["pixel_values"]         = pv
        if pa is not None: out["pixel_attention_mask"] = pa

        for key, text in [("chosen", chosen), ("rejected", rejected)]:
            resp_ids = self.processor.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_RESP,
                add_special_tokens=False,
            )["input_ids"][0]
            full_ids  = torch.cat([prompt_ids, resp_ids])[:MAX_SEQ]
            full_mask = torch.cat([prompt_mask, torch.ones_like(resp_ids)])[:MAX_SEQ]
            resp_mask = torch.cat([
                torch.zeros(len(prompt_ids), dtype=torch.long),
                torch.ones(len(resp_ids),    dtype=torch.long),
            ])[:MAX_SEQ]
            out[f"{key}_input_ids"]      = full_ids.unsqueeze(0)
            out[f"{key}_attention_mask"] = full_mask.unsqueeze(0)
            out[f"{key}_response_mask"]  = resp_mask.unsqueeze(0)

        return out


def get_sequence_logps(model, input_ids, attention_mask, response_mask,
                       pixel_values=None, pixel_attention_mask=None):
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if pixel_values         is not None: kwargs["pixel_values"]         = pixel_values
    if pixel_attention_mask is not None: kwargs["pixel_attention_mask"] = pixel_attention_mask
    logits      = model(**kwargs).logits
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
    if pv is not None: kw["pixel_values"]         = pv.to(device)
    if pa is not None: kw["pixel_attention_mask"]  = pa.to(device)
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
            try:
                kw = _pixel_kwargs(batch, device)
                c  = _seq_kwargs(batch, "chosen",   device)
                r  = _seq_kwargs(batch, "rejected",  device)
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


def load_base_model():
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    peft_model = PeftModel.from_pretrained(base, SFT_CHECKPOINT)
    return peft_model.merge_and_unload()

logger.info("Loading policy model (checkpoint-800 merged)...")
model = load_base_model()

logger.info("Loading reference model (frozen)...")
model_ref = load_base_model()
model_ref.eval()
for p in model_ref.parameters():
    p.requires_grad = False

lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

DEVICE = next(model.parameters()).device
os.makedirs(OUTPUT_DIR, exist_ok=True)

collator     = RLAIFCollator(processor)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True,  collate_fn=collator, num_workers=0)
eval_loader  = DataLoader(val_data,   batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

total_steps    = len(train_loader) // GRAD_ACCUM
optimizer      = AdamW(model.parameters(), lr=LR)
scheduler      = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)

model.train()
optimizer.zero_grad()
global_step  = 0
running_loss = 0.0
best_eval    = float("inf")

for i, batch in enumerate(train_loader):
    try:
        kw = _pixel_kwargs(batch, DEVICE)
        c  = _seq_kwargs(batch, "chosen",   DEVICE)
        r  = _seq_kwargs(batch, "rejected",  DEVICE)

        pol_c = get_sequence_logps(model, **c, **kw)
        pol_r = get_sequence_logps(model, **r, **kw)
        with torch.no_grad():
            ref_c = get_sequence_logps(model_ref, **c, **kw)
            ref_r = get_sequence_logps(model_ref, **r, **kw)

        loss = -F.logsigmoid(BETA * ((pol_c - ref_c) - (pol_r - ref_r))).mean()
        loss = loss / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM

    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM at batch {i} — skipping")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        continue
    except Exception as e:
        logger.warning(f"Skipping batch {i}: {e}")
        continue

    if (i + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        running_loss = 0.0

        if global_step % LOG_STEPS == 0:
            logger.info(f"step={global_step}/{total_steps}  loss={running_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if global_step % EVAL_STEPS == 0:
            eval_loss = run_eval(model, model_ref, eval_loader, DEVICE)
            logger.info(f"step={global_step}  eval_loss={eval_loss:.4f}  best={best_eval:.4f}")
            if eval_loss < best_eval:
                best_eval = eval_loss
                model.save_pretrained(f"{OUTPUT_DIR}/best")
                processor.save_pretrained(f"{OUTPUT_DIR}/best")
                logger.info("  -> new best saved")
            model.save_pretrained(f"{OUTPUT_DIR}/checkpoint-{global_step}")
            logger.info(f"Checkpoint saved: {OUTPUT_DIR}/checkpoint-{global_step}")

logger.info(f"Final model saved. Best eval loss: {best_eval:.4f}")
model.save_pretrained(f"{OUTPUT_DIR}/final")
logger.info("=== RLAIF-V DPO COMPLETE === " + datetime.now().isoformat())
