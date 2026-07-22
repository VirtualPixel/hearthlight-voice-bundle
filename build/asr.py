"""Work out what the reference clip actually says, or refuse to clone from it.

GPT-SoVITS conditions on the reference clip AND on a transcript of it (see render.py). The desktop path
gets that transcript from the sidecar's Whisper. A stranger on a rented box has no sidecar and no Python,
so without this module the whole zero-setup tier renders eight hundredths of a second of noise per line.

The engine is sherpa-onnx with whisper tiny.en. The ASR spike ran three architectures at three sizes over
eleven real speaker references plus four synthetic hard cases, then fed every transcript to the real
fine-tuned model: tiny.en was the only candidate that never once collapsed the TTS across 30 sentence
renders, at 0.122 WER against faster-whisper base and 0.26 to 0.34s per clip on two cores. small.en scored
WORSE and ran 5 to 10 times slower, because this corpus is unscripted voice chat with crosstalk and the
larger English models reconstruct a fluent sentence that was never said. Do not "upgrade" the model.

REFUSAL SURVIVES, ON A NEW TRIGGER. render.py used to refuse when the caller sent no transcript. Now it
refuses when the words cannot be established honestly, which is the rule that actually matters: six of the
eleven speaker references on the author's box were being prompted with a DIFFERENT utterance's words, and
a wrong-but-grammatical prompt collapsed the render below 0.6x on 9 of 30 tries.

The model's own confidence cannot carry that decision. whisper.cpp per-token probability is anti-correlated
with quality here: the single worst clip in the corpus, 4.6 seconds of mumbling reduced to two words,
scored 0.999, higher than any clip with a correct transcript, because whisper is sure about the words it
did emit and has no way to say it dropped the rest. sherpa returns empty ys_log_probs for whisper. So the
two signals below are behavioural rather than probabilistic, and both were measured against the spike's
four genuinely-unusable clips.
"""

import hashlib
import json
import os
import struct
import wave

# Bump when the model, the gate thresholds, or the normalisation change, so old verdicts stop being reused.
CACHE_VERSION = 1

# A talking human sits at 2.0 to 4.5 words per second across every clean clip in the corpus. The two
# degenerate ones sat at 1.30 and 0.43, which is the "the model gave up and returned a fragment" case, and
# it is exactly the case that collapses GPT-SoVITS hardest. Free to compute, no false alarms on the corpus.
MIN_WORDS_PER_SEC = 1.5

# Second opinion from an architecturally different model. Above this the two transcripts are not describing
# the same audio, which on the corpus meant the audio had no recoverable words in it.
MAX_DISAGREEMENT = 0.55

# Below this there is not enough audio to establish anything, and the word-rate signal stops meaning
# anything either: a 0.7s clip can carry two words honestly and still score as a fragment.
MIN_SECONDS = 1.0


class Refused(Exception):
    """The clip's words could not be established. Clone nothing from it.

    A verdict about the AUDIO is cached, because it will be the same verdict an hour from now. A verdict
    about the BUNDLE is not: a missing model file is a packaging fault that the next update fixes, and a
    cache keyed on the clip's bytes would go on refusing that clip forever after the fix landed.
    """

    def __init__(self, msg, about_audio=True):
        Exception.__init__(self, msg)
        self.about_audio = about_audio


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_16k_mono(path):
    """Whisper wants 16kHz float32 mono. The voice pipeline is 48kHz s16 throughout, so this resamples."""
    import numpy as np

    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise Refused("reference clip is %d-byte samples, expected 16-bit" % w.getsampwidth())
        rate = w.getframerate()
        chans = w.getnchannels()
        data = w.readframes(w.getnframes())

    a = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if chans > 1:
        a = a.reshape(-1, chans).mean(axis=1)
    if rate != 16000:
        import soxr

        a = soxr.resample(a, rate, 16000)
    return np.ascontiguousarray(a, dtype=np.float32)


def _words(text):
    out = []
    for raw in text.lower().split():
        w = "".join(c for c in raw if c.isalnum())
        if w:
            out.append(w)
    return out


def _wer(ref, hyp):
    """Levenshtein over words, normalised by the reference length, same metric the spike's table used."""
    r, h = _words(ref), _words(hyp)
    if not r:
        return 1.0 if h else 0.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)] / float(len(r))


def _whisper(asr_dir, threads):
    import sherpa_onnx

    enc = os.path.join(asr_dir, "tiny.en-encoder.onnx")
    dec = os.path.join(asr_dir, "tiny.en-decoder.onnx")
    tok = os.path.join(asr_dir, "tiny.en-tokens.txt")
    for p in (enc, dec, tok):
        if not os.path.isfile(p):
            raise Refused("the bundle has no speech-to-text model, %s is missing" % os.path.basename(p),
                          about_audio=False)
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=enc, decoder=dec, tokens=tok, num_threads=threads, language="en", task="transcribe"
    )


def _zipformer(asr_dir, threads):
    """The disagreement gate's second opinion, or None when the bundle chose not to carry it.

    It is a poor transcriber on purpose: uppercase, unpunctuated, and it hears "Hey Justin" as
    "HE JUST THEN", so its text is never used as prompt text. What it is good for is being wrong in
    different places than whisper, and 27MB of transducer catches the two unusable clips that the word-rate
    floor alone lets through.
    """
    import sherpa_onnx

    enc = os.path.join(asr_dir, "gate-encoder.int8.onnx")
    dec = os.path.join(asr_dir, "gate-decoder.int8.onnx")
    joi = os.path.join(asr_dir, "gate-joiner.int8.onnx")
    tok = os.path.join(asr_dir, "gate-tokens.txt")
    if not all(os.path.isfile(p) for p in (enc, dec, joi, tok)):
        return None
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=enc, decoder=dec, joiner=joi, tokens=tok, num_threads=threads
    )


def _decode(rec, audio):
    s = rec.create_stream()
    s.accept_waveform(16000, audio)
    rec.decode_stream(s)
    return s.result.text.strip()


def _derive(wav_path, asr_dir, threads):
    """Transcribe once, then decide whether the result is worth cloning from."""
    audio = read_16k_mono(wav_path)
    seconds = len(audio) / 16000.0
    if seconds < MIN_SECONDS:
        raise Refused("reference clip is %.2fs, too little audio to establish any words" % seconds)

    text = _decode(_whisper(asr_dir, threads), audio)
    if not text:
        raise Refused("speech-to-text found no words in the reference clip")

    wps = len(_words(text)) / seconds
    if wps < MIN_WORDS_PER_SEC:
        raise Refused(
            "reference clip yields %.2f words/sec, below the %.2f floor: the transcript is a fragment, "
            "not the utterance" % (wps, MIN_WORDS_PER_SEC)
        )

    gate = _zipformer(asr_dir, threads)
    disagree = None
    if gate is not None:
        disagree = _wer(text, _decode(gate, audio))
        if disagree > MAX_DISAGREEMENT:
            raise Refused(
                "two independent models disagree on the reference clip by %.2f, above %.2f: the audio has "
                "no recoverable words" % (disagree, MAX_DISAGREEMENT)
            )
    return {"text": text, "seconds": round(seconds, 2), "wps": round(wps, 2),
            "disagree": None if disagree is None else round(disagree, 2)}


def _cache_path(cache_dir, wav_path):
    return os.path.join(cache_dir, "%s.v%d.json" % (_sha1(wav_path), CACHE_VERSION))


def transcribe_reference(wav_path, asr_dir, cache_dir, threads):
    """The transcript for this clip, from cache when we have already heard it.

    Keyed on the CONTENT of the wav, not its path, because the caller writes the pinned clip to the same
    filename every batch and re-pins a different clip into it whenever the speaker's best reference changes.
    A path key would serve a stale transcript for new audio, which is the wrong-prompt bug this module
    exists to end. Refusals are cached alongside successes: a clip that has no recoverable words still has
    none an hour later, and re-deciding that per batch would spend a second to reach the same answer.

    Raises Refused. The caller turns that into a non-zero exit, and the mod falls back to replaying the
    keeper's own recorded clips, which is a real voice saying real words.
    """
    entry_path = None
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            entry_path = _cache_path(cache_dir, wav_path)
        except OSError:
            entry_path = None

    if entry_path and os.path.isfile(entry_path):
        try:
            with open(entry_path, "r") as f:
                hit = json.load(f)
            if hit.get("refused"):
                raise Refused(hit["refused"] + " (cached)")
            if hit.get("text"):
                return hit["text"], "cached"
        except (ValueError, OSError):
            pass  # A truncated cache entry is not worth a failure; redo the work.

    try:
        got = _derive(wav_path, asr_dir, threads)
    except Refused as e:
        if e.about_audio:
            _write_cache(entry_path, {"refused": str(e)})
        raise
    _write_cache(entry_path, got)
    return got["text"], "%.2fs of audio, %.2f words/sec%s" % (
        got["seconds"], got["wps"],
        ", no second opinion in this bundle" if got["disagree"] is None
        else ", %.2f disagreement" % got["disagree"])


def _write_cache(entry_path, payload):
    if not entry_path:
        return
    try:
        tmp = entry_path + ".part"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, entry_path)
    except OSError:
        pass  # A read-only or full volume costs a second next time, nothing more.


def transcribe_isolated(wav_path, asr_dir, cache_dir, threads):
    """Same answer, in a child process, so its 370MB goes back to the host before the engine loads.

    Phase 0 established that nothing in the render process ever frees anything: unload_character reports
    success and reclaims zero, and the batch already peaks around 5.4GB. Adding ASR to that process would
    add its resident set to the peak permanently, on the 2GB-to-4GB boxes this tier exists for. A fork that
    exits hands all of it back, and the fork happens before onnxruntime or the engine is imported, so the
    child is a bare interpreter with numpy in it.
    """
    if not hasattr(os, "fork"):
        return transcribe_reference(wav_path, asr_dir, cache_dir, threads)

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.close(r)
            try:
                text, how = transcribe_reference(wav_path, asr_dir, cache_dir, threads)
                payload = json.dumps({"text": text, "how": how})
            except Refused as e:
                payload = json.dumps({"refused": str(e), "about_audio": e.about_audio})
            except Exception as e:  # noqa: BLE001 - any failure here must reach the parent as a refusal
                payload = json.dumps({"refused": "%s: %s" % (e.__class__.__name__, e)})
            blob = payload.encode("utf-8")
            os.write(w, struct.pack("<I", len(blob)) + blob)
            os.close(w)
        except Exception:  # noqa: BLE001
            code = 1
        finally:
            os._exit(code)

    os.close(w)
    buf = b""
    try:
        with os.fdopen(r, "rb") as f:
            head = f.read(4)
            if len(head) == 4:
                buf = f.read(struct.unpack("<I", head)[0])
    finally:
        os.waitpid(pid, 0)

    if not buf:
        raise Refused("the transcriber died before it could answer")
    got = json.loads(buf.decode("utf-8"))
    if got.get("refused"):
        raise Refused(got["refused"], got.get("about_audio", True))
    return got["text"], got["how"]
