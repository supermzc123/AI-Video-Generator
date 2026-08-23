# H3 Motion Context integration

This document records the inspected contract of
[`NikoDemon80/ComfyUI-H3-Motion-Context`](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
at tag `v0.3.0`, commit
`658ba11ae91737391a247cf9758d0063c43491b3` (2026-08-11).

The plugin is not currently installed in the configured ComfyUI Worker. Do not
advertise Motion Context capability until the post-restart `object_info` check
described below succeeds.

## Node contract

The package registers five nodes:

| Node type | Required inputs | Optional inputs | Outputs |
|---|---|---|---|
| `MiniMaxH3MotionContext` | `conditioning: CONDITIONING`, `vae: VAE`, `latent: LATENT`, `context_length: 5/22/39/56`, `audio_context_length: INT 0..240` | `context_frames: IMAGE`, `context_latent: LATENT`, `audio_vae: VAE`, `context_audio: AUDIO` | `conditioning: CONDITIONING`, `trim_frames: INT` |
| `MiniMaxH3MotionContextTrim` | `images: IMAGE`, `trim_frames: INT` | `audio: AUDIO`, `fps: FLOAT`, `match_tail: BOOLEAN` | `images: IMAGE`, `audio: AUDIO` |
| `MiniMaxH3MotionContextSaveLatent` | `latent: LATENT`, `filename_prefix: STRING`, `clip_index: INT` | none | `latent_path: STRING` |
| `MiniMaxH3MotionContextLoadLatent` | `latent_path: STRING`, `clip_index: INT` | none | `LATENT` |
| `MiniMaxH3MotionContextSeamProbe` | `clip_b_untrimmed: AUDIO`, `trim_frames: INT` | `clip_a_latent: LATENT`, `audio_vae: VAE`, `fps`, `window_ms`, `search_ms` | unchanged `audio: AUDIO`, `report: STRING` |

The core node adds the previous clip's tail as never-denoised keyframe rows at
the head of the next clip. Its `trim_frames` output must drive the Trim node;
both decoded picture and decoded audio must pass through Trim before export.

In API-format JSON, `context_length` is a ComfyUI combo and must be serialized
as a string such as `"22"`. Domain frame budgeting remains integer-based; the
workflow compiler performs the final schema-aware conversion.

The preferred handoff artifact is a `safetensors` file containing CPU tensors
named `video` and `audio`, with metadata
`format=h3_motion_context_av_v1`. The Load node returns a deliberately
non-decodable `{"samples": [video, audio]}` wrapper intended only for the core
node's `context_latent` input. Stock ComfyUI Save/Load Latent is incompatible
with this paired AV representation.

## API workflow profiles

An initial segment and a continuation segment require different executable API
graphs. The core node rejects a run with neither `context_latent` nor
`context_frames` connected.

Initial segment:

```text
stock H3 conditioning -> guider -> SamplerCustomAdvanced
                                  |-> video/audio decode -> ordinary output
                                  `-> MiniMaxH3MotionContextSaveLatent
```

Continuation segment:

```text
MiniMaxH3MotionContextLoadLatent.context -> MotionContext.context_latent
stock H3 conditioning -------------------> MotionContext.conditioning
stock H3 generation latent --------------> MotionContext.latent
MotionContext.conditioning -> guider -> SamplerCustomAdvanced
MotionContext.trim_frames ----------------> MotionContextTrim.trim_frames
Sampler output -> SaveLatent for the next segment
Sampler output -> video/audio decode -> MotionContextTrim -> saved clip
```

Use an exact predecessor file path for `LoadLatent`. Do not use
`clip_index=0` (load newest) for controlled execution: after a rejected re-roll,
"newest" can be the rejected segment itself. Give every task attempt a unique
`filename_prefix`; fixed-slot saves are not atomic and must not overwrite the
last approved artifact. The Worker must hash and register the resulting file as
a normal artifact instead of treating a mutable output-folder path as identity.
The Save node does not return a `ui` history payload, so remote execution needs
an explicit expected path/result receipt rather than relying on ComfyUI history
media extraction.

## Runtime constraints

- H3 runs at 24fps. The plugin hard-codes that rate for video/audio grid math;
  a Motion Context profile must reject another generation fps even though the
  standalone Trim widget accepts a configurable fps.
- Prefer `context_length=22` and `audio_context_length=24`. The former consumes
  0.9167 seconds of the sampled head; the latter carries exactly one second of
  audio. `audio_context_length=0` means "follow video length", not "disable
  audio".
- Latent handoff requires identical resolution and video latent channel count.
  It cannot resize. A resolution or model-family change starts a new chain.
- The latent path carries picture and audio together. It cannot satisfy
  `visual=false, audio=true` or `visual=true, audio=false`. Visual-only
  continuation can use decoded `context_frames` without audio, but loses the
  round-trip-free path. The current generic `IncomingContext` flags therefore
  need provider-specific validation.
- Sample duration includes the pinned head. At 124 sampled frames and a
  22-frame context, only 102 frames (4.25 seconds) remain after trimming. With
  the application's 15-second cap and H3's `17k+5` grid, the largest accepted
  sample is 345 frames; a 22-frame continuation yields 323 visible frames.
- Static text and reference conditioning may still be encoded before loading
  the diffusion model. The predecessor AV latent is a dynamic dependency and
  is attached by this node only after the prior generation finishes.
- The upstream example is a ComfyUI UI workflow, not API-format JSON. It also
  contains generic `LoraLoaderModelOnly` and Spectrum nodes, so it must not be
  registered as the production Turbo profile.

## Turbo and SageAttention compatibility

No symbol-level conflict was found with the pinned official Turbo plugin
(`Larryvrh/ComfyUI-MiniMax-H3-Turbo`, commit
`4274783a23afcfdbea3b4876cb79effd6c510785`) or the loaded KJNodes H3
SageAttention node:

- Motion Context patches `comfy.ldm.minimax.model.PackedLayout.__init__` and
  `comfy.model_base.MiniMaxH3.extra_conds` lazily on its first run.
- Official Turbo provides a sampler and modifies model weights, forward hooks,
  AdaLN object patches and diffusion-model wrappers. It does not replace either
  Motion Context target.
- KJNodes clones the model and patches each
  `diffusion_model.blocks.*.attn.forward`. It does not replace either Motion
  Context target.

Use this model path for the controlled profile:

```text
base H3 model -> MiniMaxH3TurboLoRA
              -> MiniMaxH3MemoryEfficientSageAttentionPatch
              -> guider and BasicScheduler(simple, 6 steps)
MiniMaxH3TurboSampler -> SamplerCustomAdvanced.sampler
```

Motion Context changes conditioning/layout, so it remains downstream of the
stock H3 conditioning node and upstream of the guider. Keep Spectrum and
TeaCache out of this profile. Upstream explicitly warns that Turbo and Spectrum
compound audio degradation, while the project already rejects TeaCache.

This is source-level compatibility, not a completed GPU acceptance test. The
upstream mock, node smoke, payload gate and seam-probe suites all pass in the
local ComfyUI Python environment. Its layout and payload patches also pass their
self-tests against local ComfyUI commit
`344b43989e8c56b5bb4a66cf028c834192ab59dd` in a throwaway Python process.
No installed custom-node source currently replaces the same two methods.

## Dependencies and safety

The package declares Python 3.10+ but no install dependencies. At runtime it
uses ComfyUI, PyTorch, safetensors and NumPy; optional pixel/audio fallback may
use torchaudio. These are already present in the configured ComfyUI environment.
The plugin is GPL-3.0.

Both runtime patches are process-wide after first use, but gated by private
markers so unrelated H3 graphs follow stock behavior. The plugin self-tests the
live ComfyUI layout before installing its patch and refuses foreign wrappers.
Only one Motion Context/layout patch owner may be installed. A renamed backup
inside `custom_nodes` still loads and is a conflict.

## Controlled installation and test plan

1. Clone the repository at the pinned commit into exactly one
   `custom_nodes/ComfyUI-H3-Motion-Context` directory. Record repository URL,
   commit and hashes in Worker capabilities.
2. Before restart, search all enabled custom-node Python sources for
   `PackedLayout.__init__`, `MiniMaxH3.extra_conds` and
   `motion_context_index`; stop on any other owner.
3. Restart ComfyUI once. Verify all five node types through `/object_info` and
   require their `python_module` to resolve to the pinned package. Installation
   alone is not executable capability.
4. Export separate initial and continuation API workflows from ComfyUI. Validate
   exact node IDs and typed links; do not convert the upstream UI workflow with
   string replacement.
5. Run a low-resolution, 124-frame initial segment, save a uniquely named AV
   latent, and validate its hash, metadata, keys, ranks and finite values.
6. After explicit approval, run one 124-frame continuation with 22/24 context,
   Trim both streams, then inspect duration, resolution, 32kHz audio and seam
   probe output. Free models after the batch.
7. Only after the standard continuation passes, repeat with official Turbo plus
   KJ SageAttention. Compare against standard sampling because upstream warns
   Turbo can soften picture and thicken/dull audio across a chain.

Do not retry an OOM repeatedly. Preserve the workflow, ComfyUI log, peak VRAM,
minimum free RAM, context artifact and history response, unload models, and stop
that test path for review.
