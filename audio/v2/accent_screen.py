"""
accent_screen.py - objective check that a voice really sounds like its accent label.

Renders voices.CALIBRATION_SENTENCES with every candidate piper voice and runs
an independent accent-identification model over the renders:
Jzuluaga/accent-id-commonaccent_ecapa (SpeechBrain ECAPA-TDNN trained on
Common Voice accent labels, 16 English accents incl. "indian", MIT licence).

The score per voice is the mean posterior of each accent class and the share of
sentences whose top-1 accent is the label we would assign (en-IN -> indian,
en-US -> us, en-GB -> england). This is a sanity screen, not a perception
study: a classifier trained on read Common Voice speech is a proxy for what a
listener would hear.

Runs inside .venv-speech (piper-tts + speechbrain + CPU torch):
    .venv-speech/Scripts/python.exe -m audio.v2.accent_screen \
        --out docs/research_v2/speech_accent_screen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.v2 import voices as V  # noqa: E402

ACCENT_MODEL = "Jzuluaga/accent-id-commonaccent_ecapa"
LABEL_FOR = {"en-IN": "indian", "en-US": "us", "en-GB": "england"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=None, help="voice_ids (default: all piper voices)")
    ap.add_argument("--out", default="docs/research_v2/speech_accent_screen.json")
    ap.add_argument("--n", type=int, default=len(V.CALIBRATION_SENTENCES))
    a = ap.parse_args()

    import torch
    import torchaudio
    from piper import PiperVoice, SynthesisConfig
    from speechbrain.inference.classifiers import EncoderClassifier

    from speechbrain.utils.fetching import LocalStrategy

    clf = EncoderClassifier.from_hparams(source=ACCENT_MODEL,
                                         savedir=str(V.MODELS_DIR / "accent-id-commonaccent_ecapa"),
                                         local_strategy=LocalStrategy.COPY)  # no symlinks on Windows
    labels = [clf.hparams.label_encoder.decode_ndim(i) for i in range(len(clf.hparams.label_encoder))]

    sents = V.CALIBRATION_SENTENCES[: a.n]
    results = {}
    loaded = {}
    for vid in a.voices or V.SCREEN_VOICES:
        v = V.get_voice(vid)
        if v["engine"] != "piper":
            continue
        if v["model"] not in loaded:
            loaded[v["model"]] = PiperVoice.load(str(V.ensure_piper_model(v["model"])))
        pv = loaded[v["model"]]
        sid = V.piper_speaker_id(v["model"], v["speaker"])
        probs, top = [], []
        for s in sents:
            chunks = list(pv.synthesize(s, syn_config=SynthesisConfig(speaker_id=sid)))
            x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
            sr = chunks[0].sample_rate
            wav = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).unsqueeze(0)
            out_prob, _, idx, lab = clf.classify_batch(wav)
            p = torch.softmax(out_prob[0], dim=-1).numpy() if out_prob[0].min() < 0 else out_prob[0].numpy()
            probs.append(p)
            top.append(lab[0])
        mp = np.mean(probs, axis=0)
        want = LABEL_FOR.get(v["accent"])
        ranking = sorted(zip(labels, mp.tolist()), key=lambda t: -t[1])[:4]
        results[vid] = {
            "accent_label": v["accent"], "gender": v["gender"], "status": v["status"],
            "expected_class": want,
            "mean_p_expected": float(mp[labels.index(want)]) if want in labels else None,
            "top1_share_expected": float(np.mean([t == want for t in top])),
            "top_classes": [[k, round(p, 3)] for k, p in ranking],
            "top1_counts": {k: top.count(k) for k in sorted(set(top))},
        }
        r = results[vid]
        print(f"{vid:45s} {v['accent']}  P({want})={r['mean_p_expected']:.2f}  "
              f"top1={r['top1_share_expected']:.2f}  {ranking[:3]}", flush=True)

    out = _REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": ACCENT_MODEL, "n_sentences": len(sents),
                               "classes": labels, "voices": results}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
