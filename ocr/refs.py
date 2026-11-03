"""Reference transcriptions from the publisher's own code: `infer` from the release's modeling_unlimitedocr.py on the
CPU in torch bf16, greedy, gundam tiles (crop_mode=True, the publisher's server default), written to ocr/refs/ as
<case>.txt (git-ignored). Needs torch and transformers (the release's remote code), and pages.py's pages first.

python3 ocr/refs.py MODEL_DIR CASE=page[,page...] ..."""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
torch.Tensor.cuda = lambda self, *a, **k: self  # the release's code moves masks with .cuda(); this runs on the CPU


def main(model_dir: str, cases: list[str]) -> None:
    model_dir = str(Path(model_dir).expanduser())
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16).eval()
    (HERE / "refs").mkdir(exist_ok=True)
    for case in cases:
        name, pages = case.split("=")
        files = [str(HERE / "pages" / f"{p}.png") for p in pages.split(",")]
        t = time.time()
        res = model.infer(tok, image_file=files if len(files) > 1 else files[0], crop_mode=True, return_dict=True)
        (HERE / "refs" / f"{name}.txt").write_text(res["text"])
        print(f"{name}: {len(res.get('token_ids', []))} tokens in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
