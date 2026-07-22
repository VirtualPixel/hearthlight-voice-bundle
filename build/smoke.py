"""Run one real render through a freshly frozen renderer and refuse to bless a broken one.

The job protocol is the same one VoiceRenderer.java speaks: a single JSON object on stdin, one raw
PCM file per line in the out directory, exit 0 when every line landed. ref_text is left empty on
purpose: that forces the no-transcript path, so the smoke run exercises the ASR models and the
disagreement gate on the build platform, which is exactly the part that differs between wheels.
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    exe = os.path.join(bundle, "renderer.exe" if os.name == "nt" else "renderer")
    if not os.path.exists(exe):
        print("smoke: no renderer at %s" % exe)
        return 1

    job = {
        "bundle": bundle,
        "out": out,
        "ref_wav": os.path.abspath(args.ref),
        "ref_text": "",
        "sample_rate": 48000,
        "lines": [{"key": "smoke1", "text": "The house is listening tonight."}],
    }
    p = subprocess.run([exe], input=json.dumps(job).encode(), capture_output=True, timeout=900)
    sys.stdout.write(p.stdout.decode(errors="replace"))
    sys.stderr.write(p.stderr.decode(errors="replace"))
    if p.returncode != 0:
        print("smoke: renderer exited %d" % p.returncode)
        return 1

    pcm = os.path.join(out, "smoke1.pcm")
    if not os.path.exists(pcm):
        print("smoke: no output at %s" % pcm)
        return 1
    size = os.path.getsize(pcm)
    print("smoke: %s is %d bytes" % (pcm, size))
    # 48kHz mono s16 means ~96KB per second; anything under half a second is the collapse mode the
    # renderer's own docs warn about, not a render.
    if size < 48000:
        print("smoke: output too small to be speech")
        return 1
    print("smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
