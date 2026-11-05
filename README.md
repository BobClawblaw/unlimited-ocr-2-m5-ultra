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
(`python3 -m harness run unlimited-ocr-2-m5 --plan ocr`) on the recipe's build of engine commit `0af3eb8c8`,
2026-11-05. No other model was loaded, but other work was running on the Mac (load average about 10), so these are
shared-machine numbers. Evidence: `evidence/2026-11-05-ocr/`.

| case | pages | prompt tokens | reply tokens | first token | decode | text vs reference | boxes moved |
|---|---:|---:|---:|---:|---:|---|---:|
| report-1 | 1 | 909 | 198 | 1.06 s | 260.9 tok/s | identical | 0 / 16 coordinates |
| report-2 | 1 | 909 | 141 | 0.80 s | 259.3 tok/s | identical | 0 / 16 |
| paper-1 | 1 | 909 | 543 | 0.81 s | 261.8 tok/s | identical | 5 / 44 (at most 10) |
| paper-3 | 1 | 909 | 899 | 0.80 s | 259.3 tok/s | identical | 1 / 44 (1) |
| paper-10 | 1 | 909 | 1,959 | 0.79 s | 255.5 tok/s | identical | 2 / 28 (1) |
| document (all five) | 5 | 4,525 | 3,764 | 3.90 s | 224.8 tok/s | identical | 2 / 140 (1) |

- **The pages:** two synthetic report pages and three pages of the paper that ships in the release (title page, prose, tables), made by `ocr/pages.py`.
- **The reference:** the publisher's own `infer` (the release's `modeling_unlimitedocr.py`) on the CPU in torch bf16, greedy, gundam mode, written by `ocr/refs.py`.
- **Agreement:** the transcribed text matches the reference exactly in all six cases. Some layout-box coordinates (0-999 page grid) differ: 10 of 288, mostly by 1, at most 10 on one header box. They come from bf16 sums taken in another order (CPU torch vs Metal).
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
[ashhart/TensorFold#2074](https://github.com/ashhart/TensorFold/pull/2074). The recipe pins that PR's head `0af3eb8c8`
(rebased on TensorFold main `a6e25faea`), kept as the tag `uocr-0af3eb8c8` on github.com/BobClawblaw/TensorFold:
- the SAM and CLIP vision tower, the layout of global view and tiles, and the publisher's bicubic preprocessing;
- the decoder's page2d rotary positions, DASH block-sparse attention (block keys, top-16 routing, entmax-1.5, a 128-token local window), and the MoE (64 experts, top 6, a shared expert);
- chunked prefill that is bit-identical at any cut, and decode with experts picked on the GPU (one GPU wait a token);
- the family's chat template built in, since the release ships none (`--chat-template` still overrides it);
- its 34 kernel launches registered with the kernel contract:
  `SUMMARY 34 kernels: 34 pass gates 1-4, 0 fail, 0 known findings; 31 controls: 31 red as planted, 0 missed; 0 limit problems`.

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
