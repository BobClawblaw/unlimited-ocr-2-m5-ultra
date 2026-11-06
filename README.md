# Unlimited-OCR-2 on an M5 Ultra (TensorFold native, Metal)

Baidu's [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) as released (bf16, revision `04809669`),
served by TensorFold's native Metal engine on one Mac Studio M5 Ultra. Page images go in and a structured
transcription comes out, through an OpenAI-style `/v1/chat/completions`. One or many pages make one document,
transcribed page by page with layout blocks, boxes, tables as HTML and formulas as LaTeX.

Nothing is converted or quantized. The engine reads the release's safetensors as published. Its caches are bf16
keys and values with fp32 router keys, as in the publisher's code, and decoding is greedy under the publisher's page
guard.

## What it serves

| | |
|---|---|
| Input | page images in the user message (`image_url` data URLs), in order; an optional text part replaces the default instruction `Multi page parsing.` |
| Page modes | each page in gundam mode (the 1024 global view plus 640 tiles, the publisher's server default) when its tokens fit, else base mode (the global view alone, 273 tokens) |
| Image formats | PNG and WebP. JPEG needs libjpeg-turbo 3.1.4.1 (below) |
| Context | 131,072 tokens (the publisher's `max_length`): every page's image tokens, the instruction and the reply |
| Reply | greedy, the publisher's decoding; it fills the room the pages leave unless `max_tokens` asks for less, and the page guard ends it after the last page |
| Streams | one at a time; a second request waits for the first |
| Pages per request | 64 by default (`max_pages`) |

Not offered: sampling, tools, structured output, thinking, drafts (the model has no draft head), a prompt cache
(every request brings new pages), and the publisher's optional n-gram repetition filter (off by default in its own
code too).

## Speed and agreement with the publisher's code

Measured with the [recipe harness](https://github.com/BobClawblaw/recipe-harness)
(`python3 -m harness run unlimited-ocr-2-m5 --plan ocr`) on the recipe's build of engine commit `f7c686f36`,
2026-11-06. Two other models' servers were loaded on the GPU but idle (load average about 3). Evidence:
`evidence/2026-11-06-ocr/`.

| case | pages | prompt tokens | reply tokens | first token | decode | text vs reference | boxes moved |
|---|---:|---:|---:|---:|---:|---|---:|
| report-1 | 1 | 909 | 198 | 0.54 s | 303.1 tok/s | identical | 2 / 16 coordinates (at most 2) |
| report-2 | 1 | 909 | 141 | 0.27 s | 307.0 tok/s | identical | 0 / 16 |
| paper-1 | 1 | 909 | 543 | 0.28 s | 307.3 tok/s | identical | 1 / 44 (1) |
| paper-3 | 1 | 909 | 911 | 0.28 s | 307.1 tok/s | citations written as `$ [5, 6, ...] $` | 1 / 44 (1) |
| paper-10 | 1 | 909 | 2,004 | 0.28 s | 308.2 tok/s | table headers written in LaTeX | 0 / 28 |
| document (all five) | 5 | 4,525 | 3,767 | 1.45 s | 307.7 tok/s | differs; 34 boxes against 35 | 98 / 136 (at most 824) |

- **The pages:** two synthetic report pages and three pages of the paper that ships in the release (title page, prose, tables), made by `ocr/pages.py`.
- **The reference:** the publisher's own `infer` (the release's `modeling_unlimitedocr.py`) on the CPU in torch bf16, greedy, gundam mode, written by `ocr/refs.py`.
- **Agreement:** the transcribed text matches the reference in three of the six cases; 102 of 284 layout-box coordinates (0-999 page grid) moved, most of them in the document after one box goes missing. Each departure starts where two candidate tokens tie within one bf16 step: fed the reference's own reply, the engine picks otherwise at 2 of paper-3's 898 steps (step 145 with equal bf16 logits, step 557 by 0.25) and at 1 of paper-10's 1,958 (step 67, equal logits). The reference is fragile there for torch too: its own float32 run departs from its bf16 run at token 13 on paper-3 and at 1,660 on paper-10. The previous pin (`0af3eb8c8`) matched the text in all six cases (10 of 288 coordinates moved); it met ties of the same size but landed on torch's side of them. The current engine rewrote its SAM encoder and the decoder's attention, rotary and MoE gate from the papers and measured behaviour, as TensorFold requires.
- **Every case also checks:** the reply carries one `<PAGE>` marker per page, the same request repeated gives the same reply, and two requests at once both give it too.
- **Quality suite:** none was run; the harness's chat quality suites do not apply to an OCR model, and its own benchmarks (MPDocBench, MarathonDocBench, OmniDocBench) were not run.

## Memory

About 17 GB resident with the full 131,072-token context, as measured on the running server:
- 6.3 GiB of weights;
- 66 KiB of caches a token (8.25 GiB at 131,072 tokens);
- prompt and vision scratch.

A smaller `context` shrinks the caches. `python3 -m mac up` checks that the memory is free before it starts.

## The engine

The `unlimited-ocr` family is not in TensorFold's main yet: it is pull request
[ashhart/TensorFold#2074](https://github.com/ashhart/TensorFold/pull/2074). The recipe pins its commit `f7c686f36`
(on TensorFold main `b3933ecd3`; the PR's head since adds only a doc correction), kept as the tag `uocr-f7c686f36` on
github.com/BobClawblaw/TensorFold:
- the SAM and CLIP vision tower, the layout of global view and tiles, and the publisher's bicubic preprocessing;
- the decoder's page2d rotary positions, DASH block-sparse attention (block keys, top-16 routing, entmax-1.5, a 128-token local window), and the MoE (64 experts, top 6, a shared expert);
- chunked prefill that is bit-identical at any cut, and decode with experts picked on the GPU (one GPU wait a token);
- the family's chat template built in, since the release ships none (`--chat-template` still overrides it);
- its 30 kernels registered with the kernel contract:
  `SUMMARY 30 kernels: 30 pass gates 1-4, 0 fail, 0 known findings; 30 controls: 30 red as planted, 0 missed; 0 limit problems`;
- each image-encoder step gated alone against float32 (`tf-uocr-blocks`).

## Use

Needs macOS on Apple silicon with Metal tensor units (M5), Zig 0.17.0, git and Python 3.11+.

```sh
hf download baidu/Unlimited-OCR-2 --revision 0480966938291445591f69f08a13b041c073a96a --local-dir ~/models/Unlimited-OCR-2
cp local.example.toml local.toml     # set tensorfold_src (a TensorFold clone) and zig
python3 -m mac build                 # the pinned engine, under ~/.cache/unlimited-ocr-2
python3 -m mac validate
python3 -m mac up                    # serves http://127.0.0.1:8896/v1
python3 -m mac ocr page-1.png page-2.png
python3 -m mac down
```

Other commands: `status`, `logs`, `command` (prints the serve command), `launchd --port N` (writes a login unit
serving the LAN; loading it is left to you). Settings live in `recipe.toml`; `local.toml` and upper-case environment
variables override them.

A request by hand:

```sh
curl -s http://127.0.0.1:8896/v1/chat/completions -H 'content-type: application/json' -d "{
  \"model\": \"Unlimited-OCR-2\",
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$(base64 -i page.png)\"}}]}]}"
```

### JPEG

The engine decodes JPEG with libjpeg-turbo 3.1.4.1 only, so its pixels match the reference's. With another
release (Homebrew's, say) it refuses JPEG pages and says so at start. To use JPEG pages, build
libjpeg-turbo 3.1.4.1 and set `TENSORFOLD_LIBTURBOJPEG` to its `libturbojpeg.0.dylib`, or send PNG or WebP.

## Checking it again

```sh
python3 ocr/pages.py ~/models/Unlimited-OCR-2                                     # needs Pillow and PyMuPDF 1.27.2.2
python3 ocr/refs.py ~/models/Unlimited-OCR-2 report-1=report-1 report-2=report-2 \
  paper-1=paper-1 paper-3=paper-3 paper-10=paper-10 \
  document=report-1,report-2,paper-1,paper-3,paper-10                             # torch + transformers; CPU, about 20 min
```

Then run the harness's `ocr` plan against the `unlimited-ocr-2-m5` target with `ocr_dir` pointing at `ocr/`.

## License

The recipe is MIT. The model is Baidu's, released under MIT (its model card).
