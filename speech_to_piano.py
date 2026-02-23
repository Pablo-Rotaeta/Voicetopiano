"""
speech_to_piano.py
------------------
Corpus-based concatenative synthesis.

Uses fixed-rate frame triggering (dense notes) with:
  - torchcrepe neural pitch tracking
  - Normalised multi-descriptor matching
  - No-repeat penalty to avoid stuck notes

Dependencies:
    pip install librosa soundfile scipy tqdm numpy torch torchcrepe torchaudio

Cache: delete corpus_cache_*.* to rebuild if corpus changes.
"""

import os
import glob
import pickle
import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

TARGET_FILE    = "target/speech.wav"
CORPUS_FOLDER  = "corpus/piano/*.wav"
OUTPUT_FILE    = "speech_as_piano.wav"

CACHE_FEATURES = "corpus_cache_features.npz"
CACHE_AUDIO    = "corpus_cache_audio.pkl"

SR     = 22050
N_MFCC = 13

# How often a new note triggers (seconds).
# 0.08 = ~25 notes over 2s of speech — dense enough to follow intonation.
NOTE_RATE = 0.20

# How long each note plays. Should be > NOTE_RATE for legato overlap.
NOTE_DURATION = 0.5

# Silence: frames below this RMS are skipped (no note triggered).
# Very low so only true silence is skipped.
SILENCE_THRESH = 0.003

# Matching weights — all distances are normalised before weighting
PITCH_WEIGHT    = 9.0
MFCC_WEIGHT     = 1.0
CENTROID_WEIGHT = 1.5

# No-repeat: penalise notes used in the last N frames
NO_REPEAT_WINDOW  = 1
NO_REPEAT_PENALTY = 1.0   # multiply distance of recently used notes by this

# CREPE
CREPE_CONF_THRESH = 0.40

# Pitch smoothing: number of frames. Higher = smoother melody arc.
SMOOTH_FRAMES = 5

# ==========================
# CORPUS CACHE
# ==========================

def trim_to_note(audio, sr, top_db=30):
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def extract_corpus_features(audio, sr, n_mfcc=13):
    """Features from first 300ms of a trimmed, peak-normalised piano note."""
    stable = audio[:min(len(audio), int(0.3 * sr))]
    f0     = librosa.yin(stable, fmin=librosa.note_to_hz('A1'),
                         fmax=librosa.note_to_hz('C8'), sr=sr)
    pitch  = float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
    mfcc   = librosa.feature.mfcc(y=stable, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    cent   = float(librosa.feature.spectral_centroid(y=stable, sr=sr).mean())
    return pitch, mfcc, cent


def load_corpus():
    if os.path.exists(CACHE_FEATURES) and os.path.exists(CACHE_AUDIO):
        data = np.load(CACHE_FEATURES)
        if not {"pitches", "mfccs", "centroids"}.issubset(set(data.files)):
            raise RuntimeError(
                "Stale cache — delete corpus_cache_features.npz "
                "and corpus_cache_audio.pkl and re-run."
            )
        with open(CACHE_AUDIO, "rb") as fh:
            corpus_audio = pickle.load(fh)
        print(f"Corpus cache loaded: {len(corpus_audio)} notes.")
        return data["pitches"], data["mfccs"], data["centroids"], corpus_audio

    print("Building corpus cache (runs once)...")
    corpus_files = glob.glob(CORPUS_FOLDER)
    if not corpus_files:
        raise RuntimeError(f"No corpus files found at '{CORPUS_FOLDER}'.")

    pitches   = []
    mfccs     = []
    centroids = []
    audios    = []
    skipped   = 0

    for file in tqdm(corpus_files):
        audio, _ = librosa.load(file, sr=SR)
        trimmed   = trim_to_note(audio, SR, top_db=30)
        if len(trimmed) < int(0.05 * SR):
            skipped += 1
            continue
        peak = np.max(np.abs(trimmed))
        if peak > 0:
            trimmed = trimmed / peak
        pitch, mfcc, cent = extract_corpus_features(trimmed, SR, N_MFCC)
        pitches.append(pitch)
        mfccs.append(mfcc)
        centroids.append(cent)
        audios.append(trimmed)

    print(f"Loaded {len(audios)} notes ({skipped} skipped).")
    pitches   = np.array(pitches)
    mfccs     = np.array(mfccs)
    centroids = np.array(centroids)
    np.savez(CACHE_FEATURES, pitches=pitches, mfccs=mfccs, centroids=centroids)
    with open(CACHE_AUDIO, "wb") as fh:
        pickle.dump(audios, fh)
    print("Cache saved.\n")
    return pitches, mfccs, centroids, audios


corpus_pitches, corpus_mfccs, corpus_centroids, corpus_audio = load_corpus()
log_corpus_pitches = np.log1p(corpus_pitches)

# Normalisation constants (computed once from corpus statistics)
log_pitch_std  = log_corpus_pitches.std() + 1e-9
mfcc_std       = corpus_mfccs.std(axis=0) + 1e-9
cent_std       = corpus_centroids.std() + 1e-9
corpus_mfccs_norm     = corpus_mfccs / mfcc_std
corpus_centroids_norm = corpus_centroids / cent_std

# ==========================
# LOAD SPEECH
# ==========================

print("Loading speech...")
target_audio, _ = librosa.load(TARGET_FILE, sr=SR)
duration         = len(target_audio) / SR
hop_length       = int(NOTE_RATE * SR)
frame_length     = 2048
n_frames         = int(len(target_audio) / hop_length)
print(f"Duration: {duration:.2f}s  →  {n_frames} frames at {NOTE_RATE}s rate")

# ==========================
# PITCH TRACKING (CREPE)
# ==========================

print("Tracking pitch with torchcrepe...")

try:
    import torch
    import torchcrepe

    speech_16k, _ = librosa.load(TARGET_FILE, sr=16000, mono=True)
    audio_tensor  = torch.tensor(speech_16k).unsqueeze(0)
    hop_16k       = int(16000 * 0.010)   # 10ms resolution

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
    f0_raw      = f0_raw.squeeze().numpy()
    conf_raw    = conf_raw.squeeze().numpy()
    crepe_times = np.arange(len(f0_raw)) * 0.010

    # Interpolate to our frame grid
    frame_times       = np.arange(n_frames) * NOTE_RATE
    f0_frames         = np.interp(frame_times, crepe_times, f0_raw)
    confidence_frames = np.interp(frame_times, crepe_times, conf_raw)
    use_crepe         = True
    print("torchcrepe complete.")

except ImportError:
    print("[!] torchcrepe not found — using YIN fallback.")
    use_crepe = False
    f0_frames = librosa.yin(
        target_audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=SR, frame_length=frame_length, hop_length=hop_length
    )
    confidence_frames = np.ones(n_frames) * 0.7
    n_frames = min(n_frames, len(f0_frames))

# RMS per frame
rms_frames = librosa.feature.rms(
    y=target_audio, frame_length=frame_length, hop_length=hop_length)[0]
n_frames          = min(n_frames, len(f0_frames), len(rms_frames))
f0_frames         = f0_frames[:n_frames]
confidence_frames = confidence_frames[:n_frames]
rms_frames        = rms_frames[:n_frames]

# Zero out low-confidence pitch estimates
f0_frames[confidence_frames < CREPE_CONF_THRESH] = 0.0

# Smooth pitch contour: interpolate over gaps, then running mean in log-Hz
indices = np.arange(n_frames)
voiced  = f0_frames > 0
if voiced.sum() > 1:
    log_interp   = np.interp(indices, indices[voiced],
                              np.log1p(f0_frames[voiced]))
    log_smooth   = uniform_filter1d(log_interp, size=SMOOTH_FRAMES)
    f0_smooth    = np.expm1(log_smooth)
    f0_smooth[~voiced] = 0.0   # restore silence
else:
    f0_smooth = f0_frames.copy()

# Print pitch range diagnostics
voiced_hz   = f0_smooth[f0_smooth > 0]
corp_voiced = corpus_pitches[corpus_pitches > 0]
print(f"\nPitch diagnostics:")
if len(voiced_hz) > 0:
    sp_lo, sp_hi = np.percentile(voiced_hz, [5, 95])
    print(f"  Speech:  {sp_lo:.0f}–{sp_hi:.0f} Hz "
          f"({librosa.hz_to_note(sp_lo)} – {librosa.hz_to_note(sp_hi)})")
if len(corp_voiced) > 0:
    print(f"  Corpus:  {corp_voiced.min():.0f}–{corp_voiced.max():.0f} Hz "
          f"({librosa.hz_to_note(corp_voiced.min())} – "
          f"{librosa.hz_to_note(corp_voiced.max())})")
    if len(voiced_hz) > 0:
        overlap = max(sp_lo, corp_voiced.min()) < min(sp_hi, corp_voiced.max())
        print(f"  Overlap: {'✓' if overlap else '[!] WARNING — no overlap!'}")
print()

# ==========================
# MATCH FRAMES → NOTES
# ==========================

print("Matching frames to corpus notes...")

NOTE_SAMPLES   = int(NOTE_DURATION * SR)
FADE_IN_SAMPS  = int(0.010 * SR)
FADE_OUT_SAMPS = int(0.030 * SR)

output_len    = n_frames * hop_length + NOTE_SAMPLES + SR
output_buffer = np.zeros(output_len)

recent_indices = []

for i in tqdm(range(n_frames)):
    frame_rms = float(rms_frames[i])
    frame_f0  = float(f0_smooth[i])

    # Skip true silence
    if frame_rms < SILENCE_THRESH:
        recent_indices = []   # reset no-repeat on silence
        continue

    # --- Pitch distance ---
    if frame_f0 > 0:
        pitch_dists = (np.abs(log_corpus_pitches - np.log1p(frame_f0))
                       / log_pitch_std) * PITCH_WEIGHT
    else:
        pitch_dists = np.zeros(len(corpus_pitches))

    # --- MFCC distance ---
    start_samp   = i * hop_length
    speech_slice = target_audio[start_samp:start_samp + frame_length]
    if len(speech_slice) < frame_length:
        speech_slice = np.pad(speech_slice, (0, frame_length - len(speech_slice)))

    seg_mfcc      = librosa.feature.mfcc(
        y=speech_slice, sr=SR, n_mfcc=N_MFCC).mean(axis=1)
    seg_mfcc_norm = seg_mfcc / mfcc_std
    mfcc_dists    = cdist([seg_mfcc_norm], corpus_mfccs_norm,
                           metric="euclidean")[0] * MFCC_WEIGHT

    # --- Centroid distance ---
    seg_cent      = float(librosa.feature.spectral_centroid(
        y=speech_slice, sr=SR).mean())
    cent_dists    = (np.abs(corpus_centroids_norm - seg_cent / cent_std)
                     * CENTROID_WEIGHT)

    total_dists = pitch_dists + mfcc_dists + cent_dists

    # No-repeat penalty
    for idx in recent_indices:
        total_dists[idx] *= NO_REPEAT_PENALTY

    best_index = int(np.argmin(total_dists))

    recent_indices.append(best_index)
    if len(recent_indices) > NO_REPEAT_WINDOW:
        recent_indices.pop(0)

    # --- Build grain ---
    note_audio = corpus_audio[best_index].copy()
    grain      = note_audio[:min(NOTE_SAMPLES, len(note_audio))]
    if len(grain) < NOTE_SAMPLES:
        # Pad short notes with faded tail
        repeats = int(np.ceil(NOTE_SAMPLES / len(grain)))
        grain   = np.tile(grain, repeats)[:NOTE_SAMPLES]
        grain  *= np.linspace(1.0, 0.0, NOTE_SAMPLES)

    # Scale to speech frame RMS, clamped
    grain_rms = float(np.sqrt(np.mean(grain ** 2))) + 1e-9
    scale     = np.clip(frame_rms / grain_rms, 0.05, 8.0)
    grain     = grain * scale

    # Fade in/out
    fi = min(FADE_IN_SAMPS, len(grain))
    fo = min(FADE_OUT_SAMPS, len(grain))
    grain[:fi]  *= np.linspace(0.0, 1.0, fi)
    grain[-fo:] *= np.linspace(1.0, 0.0, fo)

    # Overlap-add
    start_out = i * hop_length
    end_out   = start_out + NOTE_SAMPLES
    if end_out <= len(output_buffer):
        output_buffer[start_out:end_out] += grain

# ==========================
# RENDER
# ==========================

print("Rendering output...")
nz = np.nonzero(output_buffer)[0]
if len(nz) == 0:
    raise RuntimeError("Output is empty.")
output_buffer = output_buffer[:nz[-1] + 1]

peak = np.max(np.abs(output_buffer))
if peak > 0:
    output_buffer = output_buffer / peak * 0.9

sf.write(OUTPUT_FILE, output_buffer, SR)
print(f"\nDone! Saved as: {OUTPUT_FILE}  ({len(output_buffer)/SR:.2f}s)")
print(f"Frames processed: {n_frames}  |  Note rate: {NOTE_RATE}s")
print("\n--- Tuning guide ---")
print("  NOTE_RATE         : lower = more notes (try 0.05 for very dense)")
print("  NOTE_DURATION     : longer = more legato overlap")
print("  PITCH_WEIGHT      : higher = pitch accuracy prioritised more")
print("  NO_REPEAT_WINDOW  : how many recent notes to penalise")
print("  NO_REPEAT_PENALTY : higher = stronger push for note variety")
print("  SMOOTH_FRAMES     : higher = smoother pitch melody")
