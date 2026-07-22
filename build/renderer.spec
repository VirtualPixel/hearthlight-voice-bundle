# PyInstaller spec for the bundled renderer. Read by build-renderer.sh; HL_RENDER_PY names the source.
#
# onedir, never onefile. A onefile build extracts its whole payload to a temp directory at every launch,
# and the Phase 0 spike measured a real Pterodactyl game container capping /tmp at 100MB against a payload
# several times that. onedir extracts nothing.
#
# The excludes are most of the size. A naive --collect-all build was 1.2GB; this is 314MB. torch alone was
# 628MB of a program that never calls it, pulled in only by genie_tts's build-time ONNX conversion path,
# and inference here is pure onnxruntime. sudachidict and pyopenjtalk are the Chinese and Japanese
# frontends for an English-only tier. nltk is NOT excluded and must not be: it carries the English G2P
# tables the renderer needs on every line.

import os

from PyInstaller.utils.hooks import collect_all

SOURCE = os.environ["HL_RENDER_PY"]

datas, binaries, hiddenimports = [], [], ["numpy"]
# sherpa_onnx is the speech-to-text runtime for the no-transcript path. It is imported inside a function,
# which modulegraph follows, but its native library only ships when the package is collected whole.
for pkg in ("genie_tts", "onnxruntime", "soxr", "nltk", "sherpa_onnx"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [SOURCE],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchaudio", "torchvision",
        "pyopenjtalk", "sudachipy", "sudachidict_core",
        "transformers", "matplotlib", "scipy", "uvloop", "hf_xet",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="renderer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX off deliberately. It saves little once the payload is gzipped for shipping anyway, it costs
    # decompression on every launch of a program that is already slow to start, and a packed executable is
    # a routine antivirus false positive on the Windows hosts this will eventually reach.
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="renderer")
