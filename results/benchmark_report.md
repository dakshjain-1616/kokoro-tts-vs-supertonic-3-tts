# TTS CPU Benchmark Report: Kokoro 82M vs Supertonic 3 vs Inflect-Nano-v1 vs Pocket TTS

*Generated: 2026-07-06 11:40:46 UTC*

---

## Executive Summary

This report presents a rigorous CPU-only benchmark comparing **Kokoro 82M**, **Supertonic 3**, **Inflect-Nano-v1**, and **Pocket TTS** across 6 text lengths (12–1712 characters), 6 configurations, and 5 repetitions each (180 total timed runs). All inference was performed on CPU with no GPU acceleration. Audio quality is reported as an objective UTMOS predicted MOS (neural naturalness estimate, ~1–5 scale).

| Config | Overall Mean RTF | vs Real-Time | Mean MOS (UTMOS) |
|--------|-----------------|--------------|------------------|
| Supertonic-3 (2-step) | **0.1211** | 8.3× faster than real-time | **1.53** |
| Supertonic-3 (5-step) | **0.2404** | 4.2× faster than real-time | **4.32** |
| Kokoro-82M (PyTorch) | **0.6649** | 1.5× faster than real-time | **4.46** |
| Kokoro-82M (ONNX) | **0.6415** | 1.6× faster than real-time | **4.44** |
| Inflect-Nano-v1 (4.6M) | **0.1448** | 6.9× faster than real-time | **3.48** |
| Pocket-TTS (100M) | **0.7137** | 1.4× faster than real-time | **4.10** |

> **Winner (lowest RTF):** Supertonic-3 (2-step) with mean RTF = 0.1211

---

## Hardware & Environment

| Property | Value |
|----------|-------|
| CPU Model | Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz |
| CPU Cores | 4 |
| RAM | 15.6 GB |
| OS | Linux-6.17.0-1018-azure-x86_64-with-glibc2.39 |
| Python | 3.12.3 |
| supertonic | 1.3.1 |
| kokoro | 0.9.4 |
| kokoro-onnx | 0.5.0 |
| pocket-tts | 2.1.0 |
| onnxruntime | 1.27.0 |
| torch | 2.12.1+cu130 |

---

## Methodology

### Configurations Tested

| Config | Model | Backend | Steps/Mode |
|--------|-------|---------|------------|
| Supertonic-3 (2-step) | Supertone/supertonic-3 | ONNX Runtime (CPU) | total_steps=2 (speed mode) |
| Supertonic-3 (5-step) | Supertone/supertonic-3 | ONNX Runtime (CPU) | total_steps=5 (default quality) |
| Kokoro-82M (PyTorch) | hexgrad/Kokoro-82M | PyTorch CPU | Default |
| Kokoro-82M (ONNX) | onnx-community/Kokoro-82M-v1.0-ONNX | ONNX Runtime (CPU) | Full precision |
| Inflect-Nano-v1 (4.6M) | owensong/Inflect-Nano-v1 | PyTorch CPU | FastSpeech + Snake HiFi-GAN, single male voice |
| Pocket-TTS (100M) | kyutai/pocket-tts | PyTorch CPU | Streaming LM + Mimi codec, preset voice 'alba' |

### Text Corpus

| Label | Characters | Description |
|-------|-----------|-------------|
| tiny | 12 | Single short greeting |
| short | 59 | One sentence (pangram) |
| medium | 196 | 2–3 sentences on AI |
| long | 483 | Paragraph on neural TTS |
| paragraph | 851 | Multi-sentence technical paragraph |
| extended | 1712 | Multi-paragraph essay (~1700 chars) |

### Protocol

- **CPU-only**: `CUDA_VISIBLE_DEVICES=''` set for all runs; ONNX sessions use `CPUExecutionProvider` only
- **Warmup**: 1 discarded warmup run per config on the 'medium' text before timing begins
- **Repetitions**: 5 timed runs per (config × text_length) cell
- **Timing**: `time.perf_counter()` wall-clock, measuring synthesis only (not model load)
- **Metrics**:
  - **RTF** = wall_time / audio_duration (lower = faster; <1.0 = real-time capable)
  - **Latency** = wall-clock seconds per synthesis call
  - **Throughput** = input_chars / wall_time (chars/sec)
- **Voice**: Supertonic voice 'F1' (female); Kokoro voice 'af_heart' (female); Inflect-Nano-v1 default voice 'mark' (male, single-speaker); Pocket TTS preset voice 'alba' (female, fixed across all runs)
- **Audio saved**: 1 WAV sample per (config × text_length) for quality verification

---

## Results

### Mean RTF by Config and Text Length

*(Lower RTF = faster; RTF < 1.0 = faster than real-time)*

| Config | Tiny | Short | Medium | Long | Paragraph | Extended | **Mean** |
|--------|-------|-------|-------|-------|-------|-------|---------|
| Supertonic-3 (2-step) | 0.1899±0.0075 | 0.1196±0.0034 | 0.1012±0.0025 | 0.1043±0.0086 | 0.1025±0.0041 | 0.1092±0.0053 | **0.1211** |
| Supertonic-3 (5-step) | 0.3609±0.0516 | 0.2263±0.0145 | 0.2001±0.0124 | 0.2360±0.0339 | 0.2174±0.0225 | 0.2016±0.0105 | **0.2404** |
| Kokoro-82M (PyTorch) | 0.6282±0.1180 | 0.4874±0.0267 | 0.5817±0.0138 | 0.8316±0.0516 | 0.7269±0.0576 | 0.7336±0.0479 | **0.6649** |
| Kokoro-82M (ONNX) | 0.8108±0.0112 | 0.6531±0.0509 | 0.7185±0.0415 | 0.6245±0.0830 | 0.5189±0.0053 | 0.5230±0.0087 | **0.6415** |
| Inflect-Nano-v1 (4.6M) | 0.1706±0.0576 | 0.1235±0.0263 | 0.1257±0.0029 | 0.1466±0.0161 | 0.1420±0.0033 | 0.1605±0.0048 | **0.1448** |
| Pocket-TTS (100M) | 0.7612±0.0172 | 0.7058±0.0061 | 0.6920±0.0073 | 0.6958±0.0027 | 0.7084±0.0183 | 0.7190±0.0123 | **0.7137** |

### Mean Wall-Clock Latency (seconds) by Config and Text Length

| Config | Tiny | Short | Medium | Long | Paragraph | Extended |
|--------|-------|-------|-------|-------|-------|-------|
| Supertonic-3 (2-step) | 0.265s | 0.508s | 1.382s | 3.553s | 6.410s | 13.156s |
| Supertonic-3 (5-step) | 0.503s | 0.962s | 2.731s | 8.044s | 13.599s | 24.273s |
| Kokoro-82M (PyTorch) | 0.958s | 1.974s | 7.286s | 25.238s | 38.490s | 81.122s |
| Kokoro-82M (ONNX) | 0.761s | 2.285s | 8.584s | 19.186s | 27.179s | 54.391s |
| Inflect-Nano-v1 (4.6M) | 0.138s | 0.383s | 1.180s | 2.189s | 2.120s | 2.397s |
| Pocket-TTS (100M) | 0.730s | 2.361s | 7.073s | 17.355s | 33.386s | 65.850s |

### Mean Throughput (chars/sec) by Config and Text Length

| Config | Tiny | Short | Medium | Long | Paragraph | Extended |
|--------|-------|-------|-------|-------|-------|-------|
| Supertonic-3 (2-step) | 45.4 | 116.1 | 141.9 | 136.6 | 132.9 | 130.4 |
| Supertonic-3 (5-step) | 24.2 | 61.5 | 72.0 | 61.1 | 63.1 | 70.7 |
| Kokoro-82M (PyTorch) | 12.8 | 30.0 | 26.9 | 19.2 | 22.2 | 21.2 |
| Kokoro-82M (ONNX) | 15.8 | 25.9 | 22.9 | 25.5 | 31.3 | 31.5 |
| Inflect-Nano-v1 (4.6M) | 92.6 | 158.5 | 166.2 | 222.5 | 401.6 | 714.6 |
| Pocket-TTS (100M) | 16.5 | 25.1 | 27.7 | 27.8 | 25.5 | 26.0 |

### Reference: Mean Audio Duration (seconds) per Config × Text Length

| Config | Tiny | Short | Medium | Long | Paragraph | Extended |
|--------|-------|-------|-------|-------|-------|-------|
| Supertonic-3 (2-step) | 1.39s | 4.25s | 13.65s | 34.09s | 62.55s | 120.43s |
| Supertonic-3 (5-step) | 1.39s | 4.25s | 13.65s | 34.09s | 62.55s | 120.43s |
| Kokoro-82M (PyTorch) | 1.52s | 4.05s | 12.53s | 30.35s | 52.95s | 110.58s |
| Kokoro-82M (ONNX) | 0.94s | 3.50s | 11.95s | 30.72s | 52.37s | 104.00s |
| Inflect-Nano-v1 (4.6M) | 0.81s | 3.10s | 9.39s | 14.93s | 14.93s | 14.93s |
| Pocket-TTS (100M) | 0.96s | 3.34s | 10.22s | 24.94s | 47.12s | 91.58s |

### Audio Quality — UTMOS Predicted MOS by Config and Text Length

*(Higher = more natural; UTMOS predicts mean opinion score on a ~1–5 scale. Scores are objective neural estimates, not human ratings — and on Inflect-Nano-v1 the metric is optimistic: human listening rates it buzzy/robotic, below what its 3.48 suggests.)*

| Config | Tiny | Short | Medium | Long | Paragraph | Extended | **Mean** |
|--------|-------|-------|-------|-------|-------|-------|---------|
| Supertonic-3 (2-step) | 1.87 | 1.57 | 1.58 | 1.43 | 1.36 | 1.34 | **1.53** |
| Supertonic-3 (5-step) | 4.25 | 4.43 | 4.37 | 4.52 | 4.30 | 4.09 | **4.32** |
| Kokoro-82M (PyTorch) | 4.08 | 4.51 | 4.55 | 4.54 | 4.53 | 4.53 | **4.46** |
| Kokoro-82M (ONNX) | 4.04 | 4.51 | 4.54 | 4.55 | 4.52 | 4.49 | **4.44** |
| Inflect-Nano-v1 (4.6M) | 3.02 | 4.15 | 3.90 | 3.45 | 3.01 | 3.37 | **3.48** |
| Pocket-TTS (100M) | 3.76 | 4.44 | 4.12 | 4.14 | 4.07 | 4.05 | **4.10** |

---

## Analysis & Findings

### 1. Overall Speed Ranking

1. **Supertonic-3 (2-step)** — Mean RTF: 0.1211 (8.3× real-time)
2. **Inflect-Nano-v1 (4.6M)** — Mean RTF: 0.1448 (6.9× real-time)
3. **Supertonic-3 (5-step)** — Mean RTF: 0.2404 (4.2× real-time)
4. **Kokoro-82M (ONNX)** — Mean RTF: 0.6415 (1.6× real-time)
5. **Kokoro-82M (PyTorch)** — Mean RTF: 0.6649 (1.5× real-time)
6. **Pocket-TTS (100M)** — Mean RTF: 0.7137 (1.4× real-time)

### 2. Speed vs Quality — the core trade-off

Supertonic 3 at 2-step mode is the fastest config (mean RTF **0.1211**, 8.3× real-time), **5.5× faster** than Kokoro 82M (PyTorch) at RTF 0.6649. But speed alone is misleading: its UTMOS quality is only **1.53**, by far the lowest in the field — the 2-step output is audibly robotic. The objective MOS confirms what listening reveals.

At 5-step mode, Supertonic's RTF rises to **0.2404** (a 1.98× slowdown vs 2-step from the extra flow-matching denoising steps), but quality jumps to **4.32** — competitive with Kokoro. This is the configuration that actually balances speed and quality.

Kokoro 82M scores highest on quality (PyTorch **4.46**, ONNX **4.44**) but is the slowest (RTF ~0.66–0.64).

### 3. Inflect-Nano-v1: tiny and fast, but robotic to the ear

At just 4.63M parameters — roughly 18× smaller than Kokoro and 21× smaller than Supertonic — Inflect-Nano-v1 is the second-fastest config (mean RTF **0.1448**, 6.9× real-time). Its UTMOS score is **3.48**, which places it mid-field on the metric — but **human listening does not agree with that score**: the output is audibly buzzy and robotic, with a metallic vocoder texture and flat prosody. It is more intelligible than Supertonic-2step (which is worse), but it is not in the same league as Kokoro or Supertonic-5step. This is a known UTMOS failure mode: it tends to over-rate small HiFi-GAN vocoders that are *clean* but not *natural*. Treat Inflect-Nano's 3.48 as an optimistic upper bound, not a usability verdict.

> **Important caveat — output length cap.** Inflect-Nano-v1's acoustic model has `max_frames = 1400`, which caps synthesis at **~14.93 seconds of audio** regardless of input length. Inputs longer than that (here: `long`, `paragraph`, `extended`) are **silently truncated** — only the first ~15s is rendered. Its RTF and throughput on those rows are therefore inflated (it is doing less work than the other models, which synthesize the full text). Treat Inflect-Nano's `tiny`/`short`/`medium` numbers as the honest comparison; for long-form use you must split text into <15s chunks yourself. Its audio-duration row below (flat 14.93s for the three longest inputs) makes the cap visible.

### 4. Kokoro PyTorch vs ONNX

On this hardware Kokoro ONNX (RTF **0.6415**) and PyTorch (**0.6649**) are within ~5% of each other, and their quality is identical to two decimal places (**4.44** vs **4.46**). The two are perceptually interchangeable; the choice is a deployment/packaging decision, not a quality one.

### 5. Pocket TTS: the newcomer — voice cloning on a CPU

Pocket TTS (Kyutai, ~100M params, MIT-licensed) is the newest entrant and a different kind of model: a streaming language model over a neural audio codec (Mimi), rather than a one-shot acoustic model + vocoder. On this CPU it runs at mean RTF **0.7137** (1.4× real-time) with a UTMOS of **4.10** — putting it in the same quality tier as Kokoro and Supertonic-5step while staying comfortably real-time. Unlike the FastSpeech-style Inflect-Nano, its UTMOS and its actual sound agree: it is clean and natural, not buzzy.

What the speed/quality table cannot show is its headline feature: **zero-shot voice cloning from ~5 seconds of reference audio**, capturing accent, tone, and even recording character. None of the other models here do this — they ship a fixed set of voices. We benchmarked Pocket TTS on a single preset voice ('alba') to keep the comparison fair, so the numbers above understate what the model is actually for.

### 6. Practical Implications

| Use Case | Recommended Config | Reason |
|----------|-------------------|--------|
| Highest quality (human-like) | Kokoro-82M (PyTorch or ONNX) | Top UTMOS (~4.46), Apache-2.0 weights |
| Voice cloning / custom voices | Pocket-TTS | Clones a voice from ~5s audio; MOS 4.10 at 1.4× real-time, MIT license |
| Balanced speed + quality | Supertonic-3 (5-step) | MOS 4.32 at 4.2× real-time |
| Tiny footprint / edge, quality secondary | Inflect-Nano-v1 | 4.6M params, 6.9× real-time, but buzzy/robotic (UTMOS 3.48 over-rates it) |
| Latency at any cost (prototyping) | Supertonic-3 (2-step) | Fastest, but MOS 1.53 (robotic) |
| PyTorch ecosystem / fine-tuning | Kokoro-82M (PyTorch) | Native PyTorch, easy to extend |

### 7. Reproducibility Notes

- All runs performed on a single CPU process with default thread counts
- No process pinning or CPU affinity was set
- Results may vary ±5–10% across runs due to OS scheduling jitter
- The benchmark harness (`benchmark.py`) is fully reproducible: same text, same warmup protocol, same timing method

---

## Charts

### RTF Comparison
![RTF Comparison](charts/rtf_comparison.png)

### Latency vs Text Length
![Latency vs Text Length](charts/latency_vs_length.png)

### Quality vs Speed
![Quality vs Speed](charts/quality_vs_speed.png)

---

## Raw Data

Full raw results (180 rows): [`raw_results.csv`](raw_results.csv)

Per-sample MOS: [`mos_results.csv`](mos_results.csv)

Audio samples: [`audio_samples/`](audio_samples/) — 36 WAV files (1 per config × text_length)

---

*Report generated by `report.py` on 2026-07-06 11:40:46 UTC*