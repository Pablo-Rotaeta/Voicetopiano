# Voicetopiano
# speech_to_piano

A Python tool that converts speech recordings into piano music using **corpus-based concatenative synthesis**. Given a speech file and a library of piano note recordings, it tracks the pitch and timbre of the voice frame-by-frame and replaces each frame with the closest-matching piano note — producing an output that follows the melodic contour of the original speech.

Inspired by [AudioGuide](https://github.com/benhackbarth/audioguide), built to run on Windows.

---

## How It Works

1. **Corpus preprocessing** — Each piano note file is trimmed (silence/decay tails removed), peak-normalised, and analysed for pitch (YIN), MFCCs, and spectral centroid. Results are cached to disk so the corpus only needs to be processed once.

2. **Speech pitch tracking** — The target speech is analysed frame-by-frame using [torchcrepe](https://github.com/maxrmorrison/torchcrepe), a neural pitch tracker that handles speech much more accurately than traditional methods. Pitch is smoothed across frames to follow natural intonation arcs.

3. **Matching** — Every `NOTE_RATE` seconds, the current speech frame is matched to the best corpus note using a weighted combination of normalised pitch distance, MFCC distance, and spectral centroid distance. A no-repeat penalty prevents the same note from being chosen repeatedly.

4. **Synthesis** — Matched notes are overlap-added into an output buffer at their original timing, with short fade-in/out envelopes to prevent clicks. The output is peak-normalised and saved as a WAV file.

---

## Requirements

- Python 3.9–3.11
- Windows, macOS, or Linux

Install dependencies:

```bash
pip install librosa soundfile scipy tqdm numpy torch torchcrepe torchaudio
```

> **Note:** On first run, torchcrepe will automatically download its model weights (~72 MB). This only happens once.

---

## Project Structure

```
project/
├── speech_to_piano.py          # Main script
├── target/
│   └── speech.wav (DEMO)       # Your input speech file
├── corpus/
│   └── piano/
│       ├── note_A3.wav         # Individual piano note recordings
│       ├── note_B3.wav
│       └── ...
├── speech_as_piano.wav (DEMO)  # Main audible output
└── download_iowa_piano.py (USE TO DOWNLOAD THE CORPUS)
```

### Corpus Requirements

- One WAV file per note (individual notes work best — not phrases or chords)
- Files can be long recordings with decay tails; the script automatically trims them
- Wider pitch range = better matching across different speakers
- 100–300 notes is a good corpus size

---

## Usage

0. Use download_iowa_piano.py script to download the corpus and convert it to the right format (WAV).
1. Place your speech file at `target/speech.wav`
2. Place piano note WAVs in `corpus/piano/` (Make sure the right output from the Step 0. is in the right folder).
3. Run:

```bash
python speech_to_piano.py
```

Output is saved as `speech_as_piano.wav` in the same directory.

**Subsequent runs** load the corpus from cache and complete much faster. Delete `corpus_cache_features.npz` and `corpus_cache_audio.pkl` only if you change the corpus.

---

## Configuration

All parameters are at the top of `speech_to_piano.py`:

| Parameter | Default | Description |
|---|---|---|
| `NOTE_RATE` | `0.20` | How often a new note triggers (seconds). Lower = denser, more notes. |
| `NOTE_DURATION` | `0.5` | Max length of each played note (seconds). Should be ≥ `NOTE_RATE` for legato. |
| `SILENCE_THRESH` | `0.003` | RMS below which frames are skipped. Very low — only skips true silence. |
| `PITCH_WEIGHT` | `9.0` | How much pitch accuracy influences note matching. |
| `MFCC_WEIGHT` | `1.0` | How much timbral similarity influences matching. |
| `CENTROID_WEIGHT` | `1.5` | How much spectral brightness influences matching. |
| `NO_REPEAT_WINDOW` | `1` | Number of recent notes to penalise for reuse. |
| `NO_REPEAT_PENALTY` | `1.0` | How strongly to penalise recently used notes. |
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
  Speech:  151–239 Hz (D3 – A♯3)
  Corpus:  59–2381 Hz (A♯1 – D7)
  Overlap: ✓
```

If overlap is missing (`[!] WARNING`), the corpus doesn't contain notes close to the speaker's fundamental frequency and matching will be poor regardless of other settings. Add piano notes covering the missing range.

---

## Caching

The corpus is preprocessed and saved to two cache files:

- `corpus_cache_features.npz` — pitch, MFCC, and centroid arrays
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
