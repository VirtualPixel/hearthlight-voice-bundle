"""The bundled renderer: one batch of lines, then exit.

This program is the child process described in VoiceRenderer.java. It reads one job on stdin, writes one
raw PCM file per line, and dies. Dying is the point. The engine holds roughly 3.5GB resident for a 237MB
model and unload_character reclaims none of it (Phase 0 spike, S5), so process exit is the only reliable
free there is. Crash isolation comes along for free, which is what turns a corrupt model from a
server-killer into a quiet fall back to clip replay.

WHY THE PROMPT TEXT IS NOT OPTIONAL. GPT-SoVITS conditions on a reference clip AND on a transcript of that
clip: get_phones_and_bert runs over the prompt text and those phonemes are concatenated ahead of the target
phonemes for the autoregressive decode. There is no reference-free mode in this engine, so an empty or wrong
prompt does not degrade, it collapses. Measured on the real fine-tuned model, three real Lodger lines:

    correct transcript   2.88s  3.84s  0.68s of audio
    "hello there"        0.36s  0.32s  0.08s
    a wrong sentence     0.52s  1.72s  0.08s
    empty                0.40s  0.08s  0.08s

So the prompt is never a guess. When the caller supplies a transcript it is used verbatim, because the
desktop sidecar runs a larger Whisper than anything that fits in this bundle. When it does not, asr.py
listens to the clip and writes down what it hears, which is the only way a stranger with no sidecar and no
Python gets a usable line at all. What survives from the old contract is the refusal: a clip whose words
cannot be established is refused rather than cloned from, and the caller reads the non-zero exit as "keep
what landed, replay recorded clips for the rest". That is a real voice saying real words. Eight hundredths
of a second of noise is not, and shipping it would be worse than the silence it replaced.
"""

import json
import os
import sys
import wave

# The engine resolves its shared frontend from this before anything imports it, and the import has a side
# effect worth knowing about: with no frontend on disk, Core/Resources.py PROMPTS ON STDIN. A headless
# server would hang there forever. We are already holding stdin for the job, so point it at the bundle
# first and the prompt is never reached.
_BUNDLE = None


def fail(msg):
    print("renderer: " + msg, flush=True)
    sys.exit(2)


def cap_cpus(n):
    """Hold the renderer to n cores, by affinity rather than by asking politely.

    OMP_NUM_THREADS does nothing here: ONNX Runtime schedules on its own intra-op pool, and genie_tts
    builds the sessions itself so there is no seam to pass a thread count through. Left alone the batch
    took 1487% CPU on this rig, which on a box also running the game server is the whole machine. Affinity
    is the one lever that works from outside the library.

    It costs almost nothing. The spike measured latency as decided by sequential autoregressive decode, not
    by width: 32 threads gave a 1.05s warm median and 2 cores gave 0.91s. Narrower was not slower.
    """
    try:
        avail = len(os.sched_getaffinity(0))
        keep = max(1, min(n, avail))
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:keep]))
    except (AttributeError, OSError):
        # Not Linux, or the container already pinned us. Either way the cap is someone else's now.
        pass


def stub_speaker_encoder(frontend):
    """Keep a 184MB file out of the download that this model type never opens.

    Core/Resources.py runs ensure_exists on SV_MODEL (speaker_encoder.onnx) at import and raises when it is
    absent. It only ever checks that the path EXISTS. Speaker verification belongs to the v2Pro inference
    path, and the shipped fine-tunes are V2, so on this path the file is demanded and never read: rendering
    with a zero-byte placeholder produces byte-identical work to rendering with the real 184MB weights.

    So the bundle omits it and this puts the placeholder back, which takes 184MB off what every player
    downloads. IF A v2Pro MODEL IS EVER SHIPPED THIS MUST BE REVISITED, because that path really would load
    it and would then be reading an empty file. A real file in the bundle always wins: this only fills a gap.
    """
    sv = os.path.join(frontend, "speaker_encoder.onnx")
    if not os.path.exists(sv):
        try:
            open(sv, "ab").close()
        except OSError as e:
            fail("cannot place the speaker-encoder placeholder in the frontend: %s" % e)


def tame_onnxruntime(threads):
    """Build every ONNX session narrow and non-spinning, before the engine builds any.

    Left at its defaults ONNX Runtime opens a large intra-op pool whose idle threads BUSY-WAIT for the next
    op. Measured here: 127 threads and 1247% CPU for a three-line batch. None of it was work. Pinning the
    process to two cores with taskset did not change wall time by more than noise (7.58s against 6.94s),
    which is the tell: the cores were spinning, not computing, so taking cores away cost nothing and giving
    them back bought nothing.

    That is harmless on a dedicated box and ruinous on the colocated one this tier exists to serve, where
    those spinning threads are competing with the server tick for no throughput at all.

    genie_tts constructs its own sessions and exposes no seam to pass options through, so the options are
    installed on the class before the engine is imported. Narrow AND non-spinning: the spike found latency
    is set by sequential autoregressive decode rather than by width, so a narrow pool costs nothing real.
    """
    import onnxruntime as ort

    base = ort.InferenceSession

    class Tamed(base):
        def __init__(self, *a, **kw):
            so = kw.get("sess_options") or ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
            kw["sess_options"] = so
            super().__init__(*a, **kw)

    ort.InferenceSession = Tamed


def read_job():
    raw = sys.stdin.read()
    if not raw.strip():
        fail("empty job on stdin")
    try:
        return json.loads(raw)
    except ValueError as e:
        fail("job is not valid json: %s" % e)


def to_pcm48(wav_path, want_rate):
    """Raw mono s16 at the rate the mod's playback expects.

    The engine emits 32kHz and the voice pipeline is 48kHz throughout, so this is a real resample, not a
    header edit. soxr is already a runtime dependency of the engine, so it costs nothing to add.
    """
    with wave.open(wav_path, "rb") as w:
        if w.getsampwidth() != 2:
            fail("engine produced %d-byte samples, expected 16-bit" % w.getsampwidth())
        rate = w.getframerate()
        chans = w.getnchannels()
        data = w.readframes(w.getnframes())

    import numpy as np

    a = np.frombuffer(data, dtype="<i2")
    if chans > 1:
        a = a.reshape(-1, chans).mean(axis=1).astype(np.int16)
    if rate != want_rate:
        import soxr

        a = soxr.resample(a.astype(np.float32), rate, want_rate)
        # Clip before the cast. Resampling overshoots on transients, and a wrapped int16 is an audible
        # click on exactly the loud consonants a horror line lands on.
        a = np.clip(a, -32768.0, 32767.0).astype(np.int16)
    return a.tobytes()


def main():
    global _BUNDLE
    job = read_job()

    _BUNDLE = job.get("bundle") or os.getcwd()
    out_dir = job.get("out")
    ref_wav = job.get("ref_wav")
    ref_text = (job.get("ref_text") or "").strip()
    rate = int(job.get("sample_rate") or 48000)
    lines = job.get("lines") or []

    if not out_dir or not ref_wav:
        fail("job is missing out or ref_wav")
    if not lines:
        print("renderer: nothing to do", flush=True)
        return
    if not os.path.isfile(ref_wav):
        fail("reference clip is not there: %s" % ref_wav)

    frontend = os.path.join(_BUNDLE, "frontend")
    model = os.path.join(_BUNDLE, "base-model")
    for path, what in ((frontend, "frontend"), (model, "base-model")):
        if not os.path.isdir(path):
            fail("bundle is missing its %s at %s" % (what, path))

    os.environ["GENIE_DATA_DIR"] = frontend
    stub_speaker_encoder(frontend)
    threads = int(os.environ.get("HL_VOICE_THREADS") or 2)
    cap_cpus(threads)

    if not ref_text:
        # The caller passes a transcript on the desktop path, where the sidecar has a larger Whisper than
        # anything that fits in this bundle, and that one always wins. With no sidecar there is nobody else
        # to ask, so listen to the clip. Transcription happens ONCE per clip and is cached by content, so
        # across a batch of 236 lines this costs a third of a second in total, and nothing at all on the
        # second batch. The affinity cap above is already in force and the fork inherits it.
        import time

        import asr

        t0 = time.time()
        try:
            ref_text, how = asr.transcribe_isolated(
                ref_wav, os.environ.get("HL_VOICE_ASR") or os.path.join(_BUNDLE, "asr"),
                os.path.join(_BUNDLE, "asr-cache"), threads)
        except asr.Refused as e:
            fail(str(e))
        print("renderer: heard %r (%s, %.2fs)" % (ref_text, how, time.time() - t0), flush=True)

    tame_onnxruntime(threads)

    tmp = os.environ.get("HL_VOICE_TMP") or os.environ.get("TMPDIR") or os.path.join(_BUNDLE, "tmp")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    import genie_tts as g

    g.load_character("hl", model, "English")
    g.set_reference_audio("hl", ref_wav, ref_text, "English")

    done = 0
    for line in lines:
        key = line.get("key")
        text = (line.get("text") or "").strip()
        if not key or not text:
            continue
        stage = os.path.join(tmp, key + ".wav")
        final = os.path.join(out_dir, key + ".pcm")
        part = final + ".part"
        try:
            g.tts("hl", text, save_path=stage)
            pcm = to_pcm48(stage, rate)
            with open(part, "wb") as f:
                f.write(pcm)
            # Move into place only once the bytes are all there. The caller counts files on disk rather
            # than trusting anything this program prints, so a half-written file would be counted as a
            # rendered line and cached as one.
            os.replace(part, final)
            done += 1
            print("ok %s" % key, flush=True)
        except Exception as e:
            # One bad line is not a bad batch. The rest still render, the caller re-queues what is missing.
            for junk in (part, stage):
                try:
                    os.remove(junk)
                except OSError:
                    pass
            print("skip %s: %s: %s" % (key, e.__class__.__name__, e), flush=True)
        else:
            try:
                os.remove(stage)
            except OSError:
                pass

    print("renderer: %d/%d" % (done, len(lines)), flush=True)


if __name__ == "__main__":
    main()
