"""Dump a reference encode of the MOSS-Audio-Tokenizer v1 codec.

The delay family clones a speaker from a short reference recording, so the codec has to
run in the encode direction as well. This encodes a fixed waveform with the checkpoint's
own MossAudioTokenizerModel and writes the fixture that
tests/moss/codec_encode_parity.cpp compares against.

The waveform is generated rather than read from a file so the fixture stays reproducible
without shipping audio: a sum of three sine partials with a slow amplitude envelope, which
gives the quantizer something with structure to latch onto instead of noise.

    python3 tools/community_models/moss_delay_reference_codec_encode.py \
        --codec /path/to/audio_tokenizer \
        --output tests/moss/reference/ref_codec_v1_encode.json
"""

import argparse
import json
import math
import pathlib

import torch
from transformers import AutoModel

SAMPLE_RATE = 24000
SECONDS = 4
QUANTIZERS = 16


def waveform() -> torch.Tensor:
    length = SAMPLE_RATE * SECONDS
    t = torch.arange(length, dtype=torch.float64) / SAMPLE_RATE
    envelope = 0.5 + 0.4 * torch.sin(2 * math.pi * 0.7 * t)
    signal = (
        0.55 * torch.sin(2 * math.pi * 174.0 * t)
        + 0.28 * torch.sin(2 * math.pi * 349.0 * t + 0.6)
        + 0.13 * torch.sin(2 * math.pi * 523.0 * t + 1.3)
    )
    return (envelope * signal * 0.7).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantizers", type=int, default=QUANTIZERS)
    args = parser.parse_args()

    audio = waveform()
    codec = AutoModel.from_pretrained(args.codec, trust_remote_code=True).eval()
    with torch.no_grad():
        out = codec.encode(audio.reshape(1, 1, -1), num_quantizers=args.quantizers)
    codes = out.audio_codes if hasattr(out, "audio_codes") else out[0]
    codes = codes.squeeze().to(torch.int32)             # (num_quantizers, frames)
    if codes.shape[0] != args.quantizers:
        codes = codes.transpose(0, 1)

    payload = {
        "sample_rate": SAMPLE_RATE,
        "seconds": SECONDS,
        "quantizers": int(codes.shape[0]),
        "frames": int(codes.shape[1]),
        "codes": [[int(v) for v in row] for row in codes],
    }
    pathlib.Path(args.output).write_text(json.dumps(payload))
    print(f"{payload['quantizers']} x {payload['frames']} codes -> {args.output}")


if __name__ == "__main__":
    main()
