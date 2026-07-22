# Hearthlight voice engine bundle

Release assets for the Hearthlight Haunt local voice engine. The mod downloads
these once, on demand, into the server directory; there is nothing here to
install by hand.

Each release carries the engine renderer, its runtime payload, the shared
frontend data, the pretrained base model, a small speech-recognition model,
and the third-party license texts covering all of it. Every file's sha256 is
pinned in the mod's manifest and verified at download and again at load.

The base model is an ONNX conversion of the MIT-licensed pretrained
GPT-SoVITS weights published at huggingface.co/lj1995/GPT-SoVITS. See
third-party-licenses in each release for the full texts.
