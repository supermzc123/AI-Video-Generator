# H3 quality two-pass API workflow

This is an experimental, API-first H3 graph extracted from the quality-relevant
parts of the DaSiWa community workflow. It deliberately has no UI switches,
preview graph, watermark, pixel upscalers, interpolation, cache, or save variants.

## Fixed execution path

```text
cached H3 conditioning + base AV latent
-> tuned H3 model
-> FP16 accumulation
-> video/audio sigma shift (12/3)
-> Comfy Kitchen attention
-> Euler + simple, 8-step full-denoise sample
-> split video/audio latent
-> learned 3D video-latent upscale 1.5x
-> merge unchanged audio latent
-> Euler + simple, 8-step 0.35-denoise refinement
-> decode and save
```

This is genuine two-pass refinement. The source community graph only applies
its learned latent upscaler before its single active sampling pass; its two
samplers are alternative FL2VA and Ref2VA branches, not sequential passes.

## Director contract

`director-input.schema.json` keeps only controls that can materially affect H3:

- deterministic mode routing;
- final execution prompt;
- aligned canvas and reference-fit policy;
- ordered reference slots;
- per-reference V, A, or V+A stream selection;
- video/audio trim ranges;
- per-reference local prompt;
- preserve, allow-change, and forbid-propagation attribute lists;
- optional frozen Motion Context predecessor state.

The Director is a control-plane concern. It must compile this contract into the
existing cached conditioning workload; no Director UI node belongs in the GPU
workflow.

## Required external component

The graph requires `MinimaxH3LatentUpscaler3D` and
`minimax_h3_latent_upscaler_3d_bf16.safetensors`. They are not currently present
on the configured local ComfyUI. The graph is therefore an isolated experiment,
not a production resource or selectable project profile yet.

Before promotion, run controlled comparisons using identical prompt, references,
base seed, and canvas:

1. current production H3 output;
2. first pass only;
3. two-pass at denoise 0.25, 0.35, and 0.45;
4. inspect facial structure, hands, fine texture, temporal shimmer, motion trails,
   audio continuity, runtime, and peak VRAM.

Do not promote it if the second pass merely sharpens existing malformed anatomy
or introduces temporal shimmer. Motion Context continuation also requires a
separate high-resolution continuation graph and must be validated before this
can replace the current primary workflow.
