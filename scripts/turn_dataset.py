"""Extract a labeled turn-taking set for bench_turns from Pipecat's Smart Turn
v3.2 test data (https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test,
CC-BY-4.0). Downloads one shard into ~/.cache/chirp/turn-data and writes
<id>.pcm (16 kHz mono PCM16) plus labels.tsv for one language.

    uv run --no-project --with pyarrow --with soundfile --with numpy \
        python scripts/turn_dataset.py [shard 0-9] [language, default eng]
"""
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile

shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
language = sys.argv[2] if len(sys.argv) > 2 else "eng"
cache = Path.home() / ".cache/chirp/turn-data"
cache.mkdir(parents=True, exist_ok=True)
parquet = cache / f"shard{shard}.parquet"
if not parquet.exists():
    url = (
        "https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test/resolve/main/"
        f"data/train-{shard:05d}-of-00010.parquet"
    )
    urllib.request.urlretrieve(url, parquet)
out = cache / language
out.mkdir(exist_ok=True)
rows = pq.read_table(parquet).to_pylist()
with open(out / "labels.tsv", "w") as labels:
    labels.write("id\tcomplete\tsynthetic\tmidfiller\tendfiller\tseconds\n")
    for row in rows:
        if row["language"] != language:
            continue
        audio, rate = soundfile.read(BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != 16000:  # linear resampling is enough for a turn-taking bench
            positions = np.arange(0, len(audio), rate / 16000)
            audio = np.interp(positions, np.arange(len(audio)), audio)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        pcm.tofile(out / f"{row['id']}.pcm")
        labels.write(
            f"{row['id']}\t{int(row['endpoint_bool'])}\t{int(row['synthetic'])}\t"
            f"{int(bool(row['midfiller']))}\t{int(bool(row['endfiller']))}\t{len(pcm) / 16000:.2f}\n"
        )
print(out, sum(1 for _ in open(out / "labels.tsv")) - 1, "clips")
