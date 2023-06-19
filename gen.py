import torchaudio
import torch

import sys
import os

os.environ["MUSICGEN_ROOT"] = os.path.join("v", "cache")

import time

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write, audio_read
from audiocraft.data.audio_utils import normalize_audio, convert_audio_channels, f32_pcm

dur = 30
backlog = 20

model = MusicGen.get_pretrained("large")
model.set_generation_params(duration=dur)


def gen(txt, loops):
    print(f"generating '{txt}'")
    prompt = [txt]
    wav = model.generate(prompt)  # generates all samples.

    name = f"clip_{int(time.time())}"
    z = 0
    audio_write(f"{name}_{z:04d}", wav[0].cpu(), model.sample_rate, strategy="loudness")
    extend(name, loops, prompt)


def extend(name, loops, prompt):
    print(f"extend {name} {prompt}")
    for n in range(loops):
        print(n)
        wav, sr = audio_read(
            f"{name}_{n:04d}.wav", seek_time=backlog, duration=dur - backlog
        )
        out = model.generate_continuation(wav, sr, prompt)
        n += 1
        audio_write(f"{name}_{n:04d}", out.cpu()[-1], sr, strategy="loudness")


args = sys.argv[1:]
if len(args) == 0:
    print(f"usage: {sys.argv[0]} prompt [input]")
elif len(args) == 1:
    print("starting generation")
    gen(args[0], 15)
else:
    extend(args[1], 9000, [args[0]])
