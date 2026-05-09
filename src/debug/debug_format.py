import json, os, sys
sys.path.insert(0, "/usershome/cs671_user2/p16_blv")
from PIL import Image
from trl.data_utils import is_conversational
TRAIN_DATA = "/usershome/cs671_user2/p16_blv/data/generated/grpo_train.json"
with open(TRAIN_DATA) as f:
    data = json.load(f)[:3]
sample = data[0]
image = Image.open(sample["keyframe_path"]).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}]}]
item = {"prompt": messages, "images": [image], "gemma_reference": sample["gemma_reference"], "sample_id": sample["sample_id"]}
print("item keys:", list(item.keys()))
print("prompt type:", type(item["prompt"]))
print("prompt[0] type:", type(item["prompt"][0]))
print("prompt[0] has role:", "role" in item["prompt"][0])
print("is_conversational result:", is_conversational(item))
from trl.trainer.utils import identity
batch = identity([item])
print("batch type:", type(batch))
print("batch[0] type:", type(batch[0]))
if isinstance(batch[0], dict):
    print("batch[0] keys:", list(batch[0].keys()))
    print("is_conversational(batch[0]):", is_conversational(batch[0]))
else:
    print("batch[0] IS:", repr(batch[0])[:200])
