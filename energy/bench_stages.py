"""Per-stage energy of the local voice pipeline on this laptop.

Every stage is measured with the paired idle/active protocol in
energy/protocol.py: GPU energy from NVML power (integrated), CPU package
energy from RAPL via the Windows Energy Meter Interface, both net of the
flanking idle windows. Configurations follow the production config
(configs/config.yaml) and the v2 retrieval profiles (v2/pipeline
retrieval/profiles.py); GPU variants are measured alongside where they
exist so the CPU-vs-GPU placement choice can be priced too.

Stages (one "op" each):
  vad_utterance_cpu          Silero VAD over every 32 ms frame of a 4 s utterance
  asr_stream_cpu_int8        faster-whisper small, CPU int8 (production): partial
                             re-transcription every 400 ms (beam 1) on the
                             growing buffer, then the final pass (beam 5),
                             exactly as audio/stt.py does it
  asr_final_cpu_int8         the final pass alone (streaming overhead = difference)
  asr_stream_cuda_fp16       the streaming loop on the GPU
  embed_query_cpu            bge-small-en-v1.5 query embedding, CPU, 4 torch threads
                             (v2 `faiss` profile)
  embed_query_cuda           same on the GPU
  faiss_search_{60,10k,100k} IndexFlatL2 top-50 search, d=384
  rerank50_minilm_cuda       cross-encoder ms-marco-MiniLM-L6-v2 over 50 pairs
                             (v2 `faiss_rerank` default), and _cpu, and the
  rerank50_bgebase_cuda      heavier BAAI/bge-reranker-base
  tts_piper_inproc_{short,typical,long}  piper en_US-lessac-medium, voice
                             loaded once
  tts_piper_subprocess_typical  as audio/tts.py does it today: one
                             `python -m piper` process per clause

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.bench_stages [--trials 10] [--only a,b]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from energy.paths import CACHE, model_dir, piper_voice  # noqa: E402
from energy.protocol import run_paired  # noqa: E402
from energy.sampler import EnergyMeter  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "stages.json"
SR = 16000
UTTERANCE_TEXT = "Hi, can I check in early tomorrow morning, and is there an extra fee?"
TORCH_THREADS = 4


def split_into_clauses(text):
    """Same splitter as audio/tts.py (kept in sync by import when possible)."""
    try:
        from audio.tts import split_into_clauses as f
        return f(text)
    except Exception:
        parts = re.split(r"(?<=[.!?;:,])\s+", text.strip())
        return [p for p in parts if p]


def typical_answers():
    """Short / typical / long answers from the cached real Mistral answers.

    typical = the retrieve-path answer whose length is the median; short and
    long = the 10th and 90th length percentiles.
    """
    ans = []
    for f in sorted((REPO / "simulator" / "e3_sweep_cache" / "answers").glob("*retrieve.json")):
        a = json.loads(f.read_text(encoding="utf-8"))["answer_text"].strip()
        if a:
            ans.append(a)
    ans.sort(key=len)
    pick = lambda q: ans[int(q * (len(ans) - 1))]
    return {"short": pick(0.10), "typical": pick(0.50), "long": pick(0.90)}


def synth_utterance(voice) -> np.ndarray:
    """A ~4 s spoken question at 16 kHz, rendered by piper (not committed)."""
    from scipy.signal import resample_poly

    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(UTTERANCE_TEXT))
    sr = voice.config.sample_rate
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    y = resample_poly(x, SR, sr).astype(np.float32)
    # 0.8 s trailing silence = silence_hang_ms, as the live stream would carry
    return np.concatenate([y, np.zeros(int(0.8 * SR), dtype=np.float32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--target-s", type=float, default=5.0)
    ap.add_argument("--idle-s", type=float, default=3.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    want = lambda n: not only or n in only or any(n.startswith(o) for o in only)

    import torch

    torch.set_num_threads(TORCH_THREADS)
    cuda = torch.cuda.is_available()
    sync = (lambda: torch.cuda.synchronize()) if cuda else None

    meter = EnergyMeter(rate_hz=50).start()
    time.sleep(2)
    results = {"started": datetime.now(timezone.utc).isoformat(),
               "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                        "torch": torch.__version__, "torch_threads": TORCH_THREADS,
                        "cpu": platform.processor()},
               "device_start": meter.device_info(), "protocol": {
                   "trials": args.trials, "target_s": args.target_s, "idle_s": args.idle_s,
                   "tail_s": 0.75, "sampler_rate_hz": 50}, "stages": {}}
    if Path(args.out).exists():
        prev = json.loads(Path(args.out).read_text())
        results["stages"].update(prev.get("stages", {}))

    def record(ts, **extra):
        s = ts.summary()
        s.update(extra)
        s["device_end"] = meter.device_info()
        s["measured_at"] = datetime.now(timezone.utc).isoformat()
        results["stages"][ts.name] = s
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        f = lambda d: "n/a" if d["mean"] is None else f"{d['mean']:.4f} [{d['lo']}, {d['hi']}]"
        print(f"== {ts.name}: gpu_net {f(s['gpu_net_j'])} J | cpu_net(RAPL) {f(s['cpu_net_j'])} J"
              f" | cpu_attr {f(s['cpu_attr_j'])} J | latency {f(s['latency_s'])} s", flush=True)

    P = dict(meter=meter, n_trials=args.trials, target_s=args.target_s, idle_s=args.idle_s)

    from piper import PiperVoice
    voice = PiperVoice.load(str(piper_voice()))
    utt = synth_utterance(voice)
    utt_s = len(utt) / SR
    speech_s = utt_s - 0.8
    results["utterance"] = {"text": UTTERANCE_TEXT, "speech_s": speech_s, "total_s": utt_s}
    print(f"utterance: {speech_s:.2f} s speech + 0.8 s trailing silence")

    # ---------------- VAD ----------------
    if want("vad_utterance_cpu"):
        from silero_vad import load_silero_vad
        vad = load_silero_vad()  # TorchScript, the model torch.hub serves
        frames = [torch.from_numpy(utt[i:i + 512].copy()) for i in range(0, len(utt) - 511, 512)]

        def vad_op():
            vad.reset_states()
            with torch.no_grad():
                for f in frames:
                    vad(f, SR)
        record(run_paired(name="vad_utterance_cpu", fn=vad_op, **P),
               frames_per_op=len(frames), audio_s=utt_s)

    # ---------------- ASR ----------------
    def asr_ops(model):
        part = int(0.4 * SR)
        cuts = list(range(part, int(speech_s * SR) + 1, part))

        def stream():
            for c in cuts:
                segs, _ = model.transcribe(utt[:c][-8 * SR:], beam_size=1,
                                           word_timestamps=False, language="en")
                list(segs)
            segs, _ = model.transcribe(utt, beam_size=5, language="en")
            list(segs)

        def final():
            segs, _ = model.transcribe(utt, beam_size=5, language="en")
            list(segs)
        return stream, final, len(cuts)

    if any(want(n) for n in ("asr_stream_cpu_int8", "asr_final_cpu_int8")):
        from faster_whisper import WhisperModel
        m = WhisperModel(str(model_dir("Systran/faster-whisper-small")), device="cpu",
                         compute_type="int8")
        stream, final, n_part = asr_ops(m)
        segs, _ = m.transcribe(utt, beam_size=5, language="en")
        results["utterance"]["asr_text_cpu"] = " ".join(s.text.strip() for s in segs)
        if want("asr_stream_cpu_int8"):
            record(run_paired(name="asr_stream_cpu_int8", fn=stream, warmup=1, k=1, **P),
                   n_partials=n_part, audio_s=utt_s)
        if want("asr_final_cpu_int8"):
            record(run_paired(name="asr_final_cpu_int8", fn=final, warmup=1, k=1, **P),
                   audio_s=utt_s)
        del m
    if cuda and want("asr_stream_cuda_fp16"):
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(str(model_dir("Systran/faster-whisper-small")), device="cuda",
                             compute_type="float16")
            stream, final, n_part = asr_ops(m)
            record(run_paired(name="asr_stream_cuda_fp16", fn=stream, warmup=1, **P),
                   n_partials=n_part, audio_s=utt_s)
            if want("asr_final_cuda_fp16"):
                record(run_paired(name="asr_final_cuda_fp16", fn=final, warmup=1, **P),
                       audio_s=utt_s)
            del m
        except Exception as e:  # missing cuBLAS/cuDNN DLLs etc.
            results.setdefault("skipped", {})["asr_stream_cuda_fp16"] = repr(e)
            print("skip asr_stream_cuda_fp16:", e)

    # ---------------- retrieval ----------------
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import faiss

    q = "Can I check in early tomorrow morning, and is there an extra fee?"
    corpus = [p.read_text(encoding="utf-8") for p in sorted((REPO / "retrieval" / "corpus").glob("*.txt"))]
    for dev in (["cpu", "cuda"] if cuda else ["cpu"]):
        name = f"embed_query_{dev}"
        if want(name):
            emb = SentenceTransformer(str(model_dir("BAAI/bge-small-en-v1.5")), device=dev)
            record(run_paired(name=name, fn=lambda: emb.encode([q], convert_to_numpy=True),
                              sync=sync if dev == "cuda" else None, **P))
            del emb

    if cuda and want("gpu_wake_tail"):
        # Energy of one GPU wake-up + clock-down tail: from rest, run one tiny
        # GPU op, keep measuring until the GPU is back at rest, subtract rest
        # power over the whole window. This is paid once per burst of GPU work
        # (i.e. about once per turn when ASR/embedding run on the GPU), on top
        # of the steady-state per-op energies.
        from energy.protocol import gpu_rest_w, mean_ci, settle_gpu
        emb = SentenceTransformer(str(model_dir("BAAI/bge-small-en-v1.5")), device="cuda")
        emb.encode([q]); torch.cuda.synchronize()
        rest = gpu_rest_w(meter)
        vals, durs = [], []
        for i in range(args.trials):
            settle_gpu(meter, rest)
            r_idle = meter.idle_window(args.idle_s)
            p_rest = r_idle.gpu_j / r_idle.duration_s
            with meter.measure("gpu_wake_tail") as m:
                emb.encode([q]); torch.cuda.synchronize()
                time.sleep(0.5)
                settle_gpu(meter, rest, hold_s=1.0, timeout_s=30.0)
            r = m.result
            vals.append(r.gpu_j - p_rest * r.duration_s)
            durs.append(r.duration_s)
            print(f"  [gpu_wake_tail] trial {i + 1}: {vals[-1]:.2f} J over {durs[-1]:.1f} s", flush=True)
        results["stages"]["gpu_wake_tail"] = {
            "name": "gpu_wake_tail", "gpu_net_j": mean_ci(vals), "duration_s": mean_ci(durs),
            "total_net_j": mean_ci(vals), "rest_w_threshold": rest,
            "device_end": meter.device_info(),
            "measured_at": datetime.now(timezone.utc).isoformat()}
        Path(args.out).write_text(json.dumps(results, indent=2))
        del emb

    rng = np.random.default_rng(0)
    emb_cpu = SentenceTransformer(str(model_dir("BAAI/bge-small-en-v1.5")), device="cpu")
    qv = emb_cpu.encode([q], convert_to_numpy=True).astype(np.float32)
    corpus_vecs = emb_cpu.encode(corpus, convert_to_numpy=True).astype(np.float32)
    for label, n in (("60", None), ("10k", 10_000), ("100k", 100_000)):
        name = f"faiss_search_{label}"
        if not want(name):
            continue
        idx = faiss.IndexFlatL2(384)
        idx.add(corpus_vecs if n is None else rng.standard_normal((n, 384)).astype(np.float32))
        record(run_paired(name=name, fn=lambda: idx.search(qv, 50), **P), index_size=idx.ntotal, k=50)

    # top-50 candidates for the reranker: the 60-passage corpus padded by repetition
    cands = (corpus * 2)[:50]
    pairs = [(q, c) for c in cands]
    for label, rid in (("minilm", "cross-encoder/ms-marco-MiniLM-L6-v2"),
                       ("bgebase", "BAAI/bge-reranker-base")):
        for dev in (["cuda", "cpu"] if cuda else ["cpu"]):
            name = f"rerank50_{label}_{dev}"
            if not want(name):
                continue
            ce = CrossEncoder(str(model_dir(rid)), device=dev, max_length=512)
            record(run_paired(name=name, fn=lambda: ce.predict(pairs, batch_size=50),
                              sync=sync if dev == "cuda" else None, **P),
                   pairs=50, mean_passage_chars=float(np.mean([len(c) for c in cands])))
            del ce

    # ---------------- TTS ----------------
    answers = typical_answers()
    results["tts_answers"] = {k: {"chars": len(v), "text": v} for k, v in answers.items()}
    for label, text in answers.items():
        name = f"tts_piper_inproc_{label}"
        if want(name):
            def tts_op(text=text):
                for c in voice.synthesize(text):
                    c.audio_int16_bytes
            audio_s = sum(len(c.audio_int16_bytes) // 2 for c in voice.synthesize(text)) / voice.config.sample_rate
            record(run_paired(name=name, fn=tts_op, **P), chars=len(text), audio_s=audio_s)

    if want("tts_piper_subprocess_typical"):
        clauses = split_into_clauses(answers["typical"])
        cmd = [sys.executable, "-m", "piper", "--model", str(piper_voice()), "--output-raw"]

        def tts_sub():
            for c in clauses:
                subprocess.run(cmd, input=c.encode("utf-8"), stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, check=True)
        record(run_paired(name="tts_piper_subprocess_typical", fn=tts_sub, warmup=1, **P),
               chars=len(answers["typical"]), clauses=len(clauses))

    meter.stop()
    results["finished"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
