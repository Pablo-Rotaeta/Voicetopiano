# Voicetopiano
# speech_to_piano

A Python tool that converts speech recordings into piano music using **corpus-based concatenative synthesis**. Given a speech file and a library of piano note recordings, it tracks the pitch of the voice frame-by-frame and replaces each frame with the closest-matching piano note, producing an output that follows the melodic contour of the original speech.

Inspired by [AudioGuide](https://github.com/benhackbarth/audioguide), built to run on Windows.

---

## How It Works

1. **Corpus preprocessing** — Each piano note file is trimmed (silence/decay tails removed), peak-normalised, and analysed for pitch (YIN). Results are cached to disk so the corpus only needs to be processed once.

2. **Speech pitch tracking** — The target speech is analysed frame-by-frame using [torchcrepe](https://github.com/maxrmorrison/torchcrepe), a neural pitch tracker that handles speech much more accurately than traditional methods. Pitch is smoothed across frames to follow natural intonation arcs.

3. **Speech-range filtering** - After pitch tracking, the script computes the 5–95% pitch range of the speech signal.
Only corpus notes within that pitch range are used for matching.

This ensures:
- No extreme high/low piano notes outside the speaker’s register
- More natural melodic mapping
- Better coherence

If no corpus notes fall inside the range, the script automatically falls back to the full corpus.

4. **Matching** — Every NOTE_RATE seconds, the current speech frame is matched to the closest corpus note using log-scaled pitch distance only.

5. **Synthesis** — Matched notes are overlap-added into an output buffer at their original timing. The output is peak-normalised and saved as a WAV file.
A second file containing the speech + piano mix is also generated together with a visual representation of the pitch contour.

---

## Requirements

- Python 3.9–3.11
- Windows, macOS, or Linux

Install dependencies:

```bash
pip install librosa soundfile scipy tqdm numpy torch torchcrepe torchaudio matplotlib
```

> **Note:** On first run, torchcrepe will automatically download its model weights (~72 MB). This only happens once.

---

## Project Structure

```
project/
├── speech_to_piano.py          # Main script
├── target/
│   └── speech.wav              # Your input speech file
├── corpus/
│   └── piano/
│       ├── note_A3.wav         # Individual piano note recordings
│       ├── note_B3.wav
│       └── ...
├── speech_as_piano.wav         # Piano-only output
├── speech_plus_piano.wav       # Mixed speech + piano output
└── download_iowa_piano.py      # Optional corpus downloader
```

### Corpus Requirements

- One WAV file per note (individual notes work best — not phrases or chords)
- Files can be long recordings with decay tails; the script automatically trims them
- Wider pitch range = better matching across different speakers
- 100–300 notes is a good corpus size

---

## Usage

1. Use download_iowa_piano.py to download the corpus and convert it to WAV.
2. Place your speech file at target/speech.wav
3. Place piano note WAVs in corpus/piano/
4. Run:
```bash
python speech_to_piano.py
```

Output files:
- speech_as_piano.wav
- speech_plus_piano.wav

On every run, a high-resolution pitch contour plot is displayed, showing:
- Raw CREPE pitch (10 ms resolution)
- Smoothed pitch contour
- Highlighted speech 5–95% pitch range

Subsequent runs load the corpus from cache and complete much faster. Delete corpus_cache_features.npz and corpus_cache_audio.pkl only if you change the corpus.

---

## Configuration

All parameters are at the top of `speech_to_piano.py`:

| Parameter | Default | Description |
|---|---|---|
| `NOTE_RATE` | `0.20` | How often a new note triggers (seconds). Lower = denser, more notes. |
| `NOTE_DURATION` | `0.5` | Max length of each played note (seconds). Should be ≥ `NOTE_RATE` for legato. |
| `SILENCE_THRESH` | `0.003` | RMS below which frames are skipped. Very low — only skips true silence. |
| `PITCH_WEIGHT` | `9.0` | How much pitch accuracy influences note matching. |
| `CREPE_CONF_THRESH` | `0.40` | Minimum torchcrepe confidence to use a pitch estimate. |
| `SMOOTH_FRAMES` | `5` | Pitch smoothing window (frames). Higher = smoother melody. |

### Tuning Tips

- **Output sounds sparse / too few notes** → lower `NOTE_RATE`
- **All notes sound the same** → raise `NO_REPEAT_PENALTY` or lower `NO_REPEAT_WINDOW`
- **Pitch doesn't follow speech melody** → raise `PITCH_WEIGHT`, lower `CREPE_CONF_THRESH`
- **Clicks between notes** → increase `NOTE_DURATION` so notes overlap more
- **Pitch diagnostics show no overlap** → your corpus doesn't cover the speaker's pitch range — add notes in that register

---

## Pitch Diagnostics

On every run the script prints a pitch range comparison:

```
Pitch diagnostics:
  Speech (5–95%): 173.5–276.0 Hz (F3 – C♯4)
  Corpus: 59.1–2380.8 Hz (A♯1 – D7)
```

Only corpus notes inside the Speech 5–95% range are used for synthesis.
If zero notes fall inside the range, the script automatically uses the full corpus and prints a warning.

---

## Caching

The corpus is preprocessed and saved to two cache files:

- `corpus_cache_features.npz` — pitch array
- `corpus_cache_audio.pkl` — trimmed, normalised audio for each note

These load in under a second on subsequent runs. **Delete both files** whenever you add, remove, or replace files in the corpus folder.

---

## Limitations

- Works best with **clear, relatively dry speech** recordings. Heavy background noise confuses pitch tracking.
- The output follows the *pitch contour* of speech but not its rhythm — notes trigger at a fixed rate, not on syllable boundaries.
- Very short speech files (< 1 second) may not produce enough frames to hear the effect clearly.

---

## Dependencies

| Package | Purpose |
|---|---|
| `librosa` | Audio analysis (MFCCs, spectral centroid, onset detection) |
| `torchcrepe` | Neural pitch tracking |
| `torch` / `torchaudio` | PyTorch backend for torchcrepe |
| `soundfile` | WAV file I/O |
| `scipy` | Distance computation, signal smoothing |
| `numpy` | Array operations |
| `tqdm` | Progress bars |

---

## License

MIT
