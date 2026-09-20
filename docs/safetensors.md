# Native .safetensors support + EXL3

Support summary reviewed against master on 2026-09-18.

buun-llama-cpp loads supported Hugging Face model directories directly, including
quantized checkpoints and EXL3. You do not need to convert them to GGUF first.
The loader uses the existing model graphs, placement system, and inference
backends rather than running a separate inference engine.

Support depends on three things: the **model architecture**, the **stored
quantization format**, and the **backend** executing it. A `.safetensors`
extension alone does not establish compatibility.

## Quick start

Point `-m` at the directory containing the weights, configuration, and tokenizer:

```sh
llama-server -m /path/to/model-directory -ngl 99
```

Or download and run a supported repository directly:

```sh
llama-server -hf unsloth/Qwen3.6-27B-NVFP4 -ngl 99
```

Downloads use the Hugging Face cache and are reused on later launches. Omit the
GGUF-style `:quant` suffix: the safetensors repository identifies the quantization.
For a repository with multiple model directories, use `-hff path/to/config.json`.
If a repository contains both GGUF and safetensors, `-hff config.json` explicitly
selects native loading. `--offline` requires the files to be cached already.

Keep the complete model directory, including any `quantization_config.json`,
shard index, scale tensors, and tokenizer files. A single weight shard is not a
complete model.

## Supported model families

| Family | Native importer coverage |
|---|---|
| Qwen2 and Qwen3 | Conventional dense models |
| Qwen3.5 / Qwen3.6 / Qwen3.8 | Dense and MoE checkpoints using the Qwen3.5 configuration family |
| Qwen3-VL | Qwen3-VL configuration family |
| Qwen4 / Qwen3.8-Flash-Next | `qwen4_exp` and `qwen4_exp_text` configurations |
| DeepSeek-V4 | `deepseek_v4`; Flash has been exercised in the native-loading campaign |
| Llama and classic Mistral | `llama` and `mistral` configurations; not every derivative architecture |
| DFlash2 drafters | Supported Qwen3-based DFlash2 configurations, including EXL3 drafters |

These are importer families, not a guarantee for every checkpoint bearing those
names. Architecture-specific dimensions, tensor layouts, and auxiliary modules
are validated during loading. GGUF architecture support does not automatically
mean native safetensors support. Gemma and MLX-specific layouts are not currently
covered by these importers.

## Supported quantization formats

The table describes accepted format families and important restrictions. It is
not a promise that every producer version or every backend supports every row.
`W4A16`, for example, means 4-bit weights with 16-bit activations in the source
format; runtime kernel selection may use a different activation representation.

| Format | Supported variants / restrictions |
|---|---|
| Unquantized | BF16, F16, and F32 model tensors |
| **EXL3** | Packed trellis weights with `mul1`, `mcg`, or `3inst` codebooks and their sign/scale vectors; mixed per-tensor bit widths |
| **NVFP4** | Packed E2M1 weights with group-16 scales and global scales; supported compressed-tensors and ModelOpt layouts, including mixed NVFP4/FP8 models |
| **FP8** | E4M3 tensor, channel, 128×128 block, and supported group-32 scaling layouts; supported static and dynamic activation contracts |
| **AWQ** | INT4 GEMM-packed checkpoints with zero points; group sizes divisible by 32 and per-channel layouts |
| **GPTQ** | INT4 and INT8 packed checkpoints; INT4 act-order is supported with restrictions below |
| **AutoRound / INC** | Supported GPTQ-style INT4/INT8, compressed-tensors packed integer, and MXFP4/MXFP8 exports—not arbitrary AutoRound layouts |
| **INT8 / SmoothQuant** | Channel-scaled INT8 with supported static or dynamic activation quantization. SmoothQuant smoothing must already be folded into the checkpoint weights |
| **Packed WNA16** | Supported compressed-tensors INT4 groups divisible by 32 and symmetric INT8 group-128 layouts |
| **W4A8** | Supported group-128 INT4 layouts with INT8 or FP8 activation contracts |
| **MXFP4 / MXFP8** | Supported group-32 microscaled layouts with E8M0 scale sidecars |
| **BitsAndBytes** | Serialized NF4/FP4, including nested scale quantization, and INT8 with SCB scales; not a promise of the full dynamic LLM.int8 outlier algorithm |
| **EETQ** | INT8 weights with per-output-channel scales |
| **Quanto** | `qint4`, `qint8`, and supported E4M3 `qfloat8` layouts; not `qint2` |
| **TorchAO** | Serialized tiled INT4 and unpacked INT8 tensor subclasses; not every TorchAO tensor subclass |
| **HQQ** | Transformers axis-1 INT4 with unquantized scale/zero metadata, plus the supported ExecuTorch INT4 expert layout; not general HQQ coverage |

ModelOpt, Quark, FBGEMM, and compressed-tensors are producer/schema families,
not interchangeable numerical formats. Their supported exports map to the
formats above; recognizing a producer name does not enable all of its exports.

### Numerical and layout limitations

- Native loading can transpose or repack tensors in memory. Some adapters use
  existing ggml representations; others preserve a dedicated packed format.
  This does not create a GGUF file or guarantee bit-identical logits to another
  engine. Activation precision and reduction order also depend on the kernel.
- ModelOpt checkpoints declaring `quant_algo: NVFP4` (W4A4) are accepted using
  the W4A16 loading contract, with a warning that serialized activation scales
  are ignored. This is not a claim to reproduce the producer's W4A4 numerics.
- GPTQ act-order is limited to supported INT4 layouts. Transformed Qwen3.5
  recurrent projections do not support arbitrary act-order mappings; INT8
  act-order and AutoRound act-order exports are not accepted.
- EXL2, AQLM, SpQR, VPTQ, HIGGS, SINQ, and generic packed 2/3/5/6-bit WNA16
  formats are not supported. EXL3's bit widths are a separate representation.

## Backends, splitting, and EXL3

CUDA has the broadest native quantization coverage. Fast kernels are selected
by GPU architecture and tensor shape; support for a format does not imply
native hardware acceleration for it on every NVIDIA generation.

EXL3 supports CPU execution and CUDA execution, including layer splitting,
tensor splitting, and MoE expert caching. HIP supports standalone EXL3 on
wave32 devices, including RDNA4, and has an EXL3 expert-cache path. The
standalone GPU executor does not support HIP wave64 devices. Optimizations and
performance vary by architecture, bit width, codebook, and batch size.

HIP support is not identical to CUDA support across the table above. In
particular, scale-aware FP8/INT8, BitsAndBytes, and GPTQ act-order operations
have CUDA/CPU paths that are not general HIP execution paths. Do not infer
full-model HIP compatibility from the presence of a parser or weight type.

CPU execution/offloading also depends on the representation: reference kernels
can be much slower than GPU kernels. A mixed-format checkpoint must have a
supported execution path for every required tensor.

VBR and Turbo/TCQ remain **KV-cache** features, independent of weight format.
Supported native models use the same cache machinery as GGUF models. These
fork-specific GPU KV features require CUDA or HIP/ROCm; Metal support is not
provided. Model-specific cache and speculative-decoding restrictions still apply.

## RAM, SSD offloading, and prepared-weight caching

Native loading supports file-backed host weights and lazy access to eligible
large tables. Compatible tensors can use their original shard bytes; tensors
needing a layout change can be streamed into prepared backing files. This
allows demand paging without requiring the whole host-resident model in RAM.
It does not make SSD access as fast as RAM or VRAM.

For a supported MoE using CPU experts, the relevant options include:

```sh
llama-server -m /path/to/model-directory \
  -ngl 99 -ot 'exps=CPU' --moe-cache auto \
  --load-mode mmap --mmap-prefetch off --lazy-mode on
```

Optionally add `--repack-cache /path/to/dedicated-cache` on Linux to retain
prepared host tensors across launches instead of reconstructing them each time.
This is opt-in and can consume **tens of GiB**. Use a directory outside the source
model and leave room for the prepared weights as well as the original download.
Without it, prepared files are disposable; their temporary location is
`LLAMA_CACHE` when set, otherwise the model directory. `--check-tensors` also
verifies persistent cached payload checksums.

See the [MoE cache guidance](../README.md#choose-the-moe-cache-mode) for tuning. Report unsupported
checkpoints or loading failures with the repository/revision, full command,
build commit, backend/GPU, and error log.
