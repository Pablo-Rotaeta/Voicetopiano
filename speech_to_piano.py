"""
speech_to_piano.py
------------------

Corpus-based concatenative synthesis
WITH:

  • Speech 5–95% pitch filtering
  • High-resolution pitch visualisation restored
  • Highlighted speech pitch range
"""

import os
import glob
import pickle
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

TARGET_FILE    = "target/speech.wav"
CORPUS_FOLDER  = "corpus/piano/*.wav"
OUTPUT_FILE    = "speech_as_piano.wav"
MIX_OUTPUT_FILE = "speech_plus_piano.wav"

CACHE_FEATURES = "corpus_cache_features.npz"
CACHE_AUDIO    = "corpus_cache_audio.pkl"

SR     = 22050
NOTE_RATE = 0.20
NOTE_DURATION = 0.5
SILENCE_THRESH = 0.003

PITCH_WEIGHT = 9.0
CREPE_CONF_THRESH = 0.40
SMOOTH_FRAMES = 5

# ==========================
# CORPUS LOADING
# ==========================

def trim_to_note(audio, sr, top_db=30):
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def extract_pitch(audio, sr):
    stable = audio[:min(len(audio), int(0.3 * sr))]
    f0 = librosa.yin(
        stable,
        fmin=librosa.note_to_hz('A1'),
        fmax=librosa.note_to_hz('C8'),
        sr=sr
    )
    return float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 0.0

def load_corpus():
    if os.path.exists(CACHE_FEATURES) and os.path.exists(CACHE_AUDIO):
        data = np.load(CACHE_FEATURES)
        with open(CACHE_AUDIO, "rb") as fh:
            corpus_audio = pickle.load(fh)
        print(f"Corpus cache loaded: {len(corpus_audio)} notes.")
        return data["pitches"], corpus_audio

    print("Building corpus cache...")
    corpus_files = glob.glob(CORPUS_FOLDER)

    pitches, audios = [], []

    for file in tqdm(corpus_files):
        audio, _ = librosa.load(file, sr=SR)
        trimmed = trim_to_note(audio, SR)

        if len(trimmed) < int(0.05 * SR):
            continue

        peak = np.max(np.abs(trimmed))
        if peak > 0:
            trimmed /= peak

        pitch = extract_pitch(trimmed, SR)

        pitches.append(pitch)
        audios.append(trimmed)

    pitches = np.array(pitches)

    np.savez(CACHE_FEATURES, pitches=pitches)
    with open(CACHE_AUDIO, "wb") as fh:
        pickle.dump(audios, fh)

    return pitches, audios

corpus_pitches, corpus_audio = load_corpus()
log_corpus_pitches = np.log1p(corpus_pitches)

# ==========================
# LOAD SPEECH
# ==========================

print("Loading speech...")
target_audio, _ = librosa.load(TARGET_FILE, sr=SR)

hop_length = int(NOTE_RATE * SR)
n_frames = int(len(target_audio) / hop_length)

# ==========================
# PITCH TRACKING
# ==========================

print("Tracking pitch...")

try:
    import torch
    import torchcrepe

    speech_16k, _ = librosa.load(TARGET_FILE, sr=16000)
    audio_tensor = torch.tensor(speech_16k).unsqueeze(0)
    hop_16k = int(16000 * 0.010)

    f0_raw, conf_raw = torchcrepe.predict(
        audio_tensor,
        sample_rate=16000,
        hop_length=hop_16k,
        fmin=50.0, fmax=1000.0,
        model='full',
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=512,
        device='cpu'
    )

    f0_raw = f0_raw.squeeze().numpy()
    conf_raw = conf_raw.squeeze().numpy()
    crepe_times = np.arange(len(f0_raw)) * 0.010

    frame_times = np.arange(n_frames) * NOTE_RATE
    f0_frames = np.interp(frame_times, crepe_times, f0_raw)
    confidence_frames = np.interp(frame_times, crepe_times, conf_raw)

except ImportError:
    f0_frames = librosa.yin(
        target_audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=SR,
        hop_length=hop_length
    )
    confidence_frames = np.ones_like(f0_frames)
    crepe_times = None

f0_frames[confidence_frames < CREPE_CONF_THRESH] = 0.0

# ==========================
# SMOOTHING
# ==========================

indices = np.arange(len(f0_frames))
voiced = f0_frames > 0

if voiced.sum() > 1:
    log_interp = np.interp(indices, indices[voiced],
                           np.log1p(f0_frames[voiced]))
    log_smooth = uniform_filter1d(log_interp,
                                  size=SMOOTH_FRAMES)
    f0_smooth = np.expm1(log_smooth)
    f0_smooth[~voiced] = 0.0
else:
    f0_smooth = f0_frames.copy()

# ==========================
# PITCH DIAGNOSTICS
# ==========================

print("\nPitch diagnostics:")

voiced_vals = f0_smooth[f0_smooth > 0]

if len(voiced_vals) > 0:
    sp_lo, sp_hi = np.percentile(voiced_vals, [5, 95])
    print(f"  Speech (5–95%): {sp_lo:.1f}–{sp_hi:.1f} Hz "
          f"({librosa.hz_to_note(sp_lo)} – {librosa.hz_to_note(sp_hi)})")
else:
    sp_lo, sp_hi = 0, 0

corp_voiced = corpus_pitches[corpus_pitches > 0]
print(f"  Corpus: {corp_voiced.min():.1f}–{corp_voiced.max():.1f} Hz "
      f"({librosa.hz_to_note(corp_voiced.min())} – "
      f"{librosa.hz_to_note(corp_voiced.max())})\n")

# ==========================
# FILTER CORPUS TO SPEECH RANGE
# ==========================

if sp_hi > 0:
    valid_mask = (corpus_pitches >= sp_lo) & (corpus_pitches <= sp_hi)
else:
    valid_mask = np.ones_like(corpus_pitches, dtype=bool)

if np.sum(valid_mask) == 0:
    print("Warning: No corpus notes inside speech range. Using full corpus.")
    valid_mask = np.ones_like(corpus_pitches, dtype=bool)

filtered_pitches = corpus_pitches[valid_mask]
filtered_logs = np.log1p(filtered_pitches)
filtered_audio = [corpus_audio[i] for i in np.where(valid_mask)[0]]

print(f"Using {len(filtered_audio)} corpus notes inside speech range.\n")

# ==========================
# VISUALISATION (RESTORED)
# ==========================

user_sentence = input("\nType the sentence that was spoken:\n> ")

plt.figure(figsize=(12, 5))

if crepe_times is not None:
    plt.plot(crepe_times, f0_raw,
             alpha=0.4, label="Raw CREPE (10ms)")

time_axis = np.arange(len(f0_smooth)) * NOTE_RATE
plt.plot(time_axis, f0_smooth,
         linewidth=2, label="Smoothed contour")

if sp_hi > 0:
    plt.axhspan(sp_lo, sp_hi, alpha=0.15,
                label="Speech 5–95% range")

plt.title(f'Pitch Contour\n"{user_sentence}"')
plt.xlabel("Time (seconds)")
plt.ylabel("Pitch (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================
# SYNTHESIS
# ==========================

NOTE_SAMPLES = int(NOTE_DURATION * SR)
output_len = len(f0_smooth) * hop_length + NOTE_SAMPLES + SR
output_buffer = np.zeros(output_len)

for i in tqdm(range(len(f0_smooth))):

    frame_f0 = f0_smooth[i]
    if frame_f0 <= 0:
        continue

    pitch_dists = np.abs(filtered_logs - np.log1p(frame_f0))
    best_index = int(np.argmin(pitch_dists))
    note_audio = filtered_audio[best_index]

    grain = note_audio[:NOTE_SAMPLES]
    if len(grain) < NOTE_SAMPLES:
        grain = np.pad(grain,
                       (0, NOTE_SAMPLES - len(grain)))

    start_out = i * hop_length
    output_buffer[start_out:start_out + NOTE_SAMPLES] += grain

output_buffer /= np.max(np.abs(output_buffer)) + 1e-9
output_buffer *= 0.9

sf.write(OUTPUT_FILE, output_buffer, SR)

speech_trimmed = target_audio[:len(output_buffer)]
speech_trimmed = np.pad(
    speech_trimmed,
    (0, max(0, len(output_buffer) - len(speech_trimmed))))

mix = speech_trimmed + output_buffer
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 0.9

sf.write(MIX_OUTPUT_FILE, mix, SR)

print("Saved files:")
print("  -", OUTPUT_FILE)
print("  -", MIX_OUTPUT_FILE)
print("Done.")
