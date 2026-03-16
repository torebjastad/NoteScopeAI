import os
import glob
import json
import re
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Map piano keys back and forth
# A0 is note index 0 (lowest on standard piano), up to C8 (index 87)
# Note: MIDI note 21 is A0.
def note_name_to_index(note_name: str) -> int:
    """
    Convert a string like "A0", "C#4", "Db5" to an integer 0-87.
    A0 -> 0; C8 -> 87
    """
    note_to_semitone = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    match = re.match(r"([A-G][b#]?)(-?\d+)", note_name)
    if not match:
        raise ValueError(f"Could not parse note name: {note_name}")
    
    pitch_class, octave = match.groups()
    octave = int(octave)
    
    semitone = note_to_semitone[pitch_class]
    
    # Calculate MIDI note number. C-1 is 0. C0 is 12. C4 is 60. A0 is 21.
    midi_note = (octave + 1) * 12 + semitone
    
    # Map to our 0-87 range (A0 is 0)
    index = midi_note - 21
    
    if index < 0 or index > 87:
        raise ValueError(f"Note {note_name} is out of standard 88-key piano range (A0-C8). Index: {index}")
        
    return index

class PianoToneDataset(Dataset):
    def __init__(self, c_dir: str, n_samples: int = 8192, augment: bool = False, transform=None, verified_files: str = None):
        self.c_dir = c_dir
        self.n_samples = n_samples
        self.augment = augment
        self.transform = transform
        self.files = []
        self.labels = []

        allowlist = None
        if verified_files is not None:
            with open(verified_files, 'r') as f:
                allowlist = set(json.load(f))

        self._crawl_directory(allowlist)
        
    def _parse_label_from_filename(self, path: str) -> int:
        filename = os.path.basename(path)
        
        # Heuristic 1: Look for exact patterns like A0v1.ogg or C#4v5.ogg
        # Usually Note + Octave + optional 'v' and velocity
        match = re.search(r"([A-G][b#]?\d+)", filename)
        if match:
            note_str = match.group(1)
            try:
                return note_name_to_index(note_str)
            except ValueError:
                pass
                
        # Heuristic 2: For FSS6 format: FSS6_Royers_L1_A#5_RR1.wav
        # We already captured it above, but we can be more specific if need be.
        
        raise ValueError(f"Could not extract pitch label from filename: {filename}")
        
    def _crawl_directory(self, allowlist=None):
        extensions = ['*.wav', '*.ogg', '*.flac']
        all_paths = []
        for ext in extensions:
            # Recursive search
            all_paths.extend(glob.glob(os.path.join(self.c_dir, '**', ext), recursive=True))

        for path in all_paths:
            if allowlist is not None and path not in allowlist:
                continue

            # Skip files that don't look like single notes (e.g. rel, pedal)
            filename = os.path.basename(path).lower()
            if any(x in filename for x in ['rel', 'pedal', 'harm', 'noise', 'silence', 'pad', 'fx']):
                continue

            try:
                label = self._parse_label_from_filename(path)
                self.files.append(path)
                self.labels.append(label)
            except ValueError:
                pass # skip files we can't parse

        print(f"Found {len(self.files)} viable audio files.")

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio; output shape is [channels, time]
        waveform, sample_rate = torchaudio.load(path)
        
        # Mix down to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Convert to numpy for librosa trim
        waveform_np = waveform.numpy()[0]
        # Trim leading/trailing silence (top_db is the threshold below reference)
        # 30db is a safe threshold for most recorded piano
        trimmed_np, _ = librosa.effects.trim(waveform_np, top_db=30)
        
        # Convert back to tensor
        waveform = torch.from_numpy(trimmed_np).unsqueeze(0)
        
        # Target samples size
        if waveform.shape[1] > self.n_samples:
            # Random crop during training to add shift augmentation
            if self.augment:
                max_start = waveform.shape[1] - self.n_samples
                # Bias towards the beginning (the attack) for piano notes
                start_idx = np.random.randint(0, min(max_start, sample_rate // 2) + 1)
            else:
                start_idx = 0
            waveform = waveform[:, start_idx:start_idx + self.n_samples]
        elif waveform.shape[1] < self.n_samples:
            # Zero pad
            padding = self.n_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Add basic waveform augmentations
        if self.augment:
            # Random gain +/- 6dB
            gain = np.random.uniform(0.5, 2.0)
            waveform = waveform * gain
            
            # Add subtle white noise
            noise_amp = np.random.uniform(0, 0.005)
            noise = torch.randn_like(waveform) * noise_amp
            waveform = waveform + noise
            
            # Safe Pitch Shifting (-2 to +2 semitones max to preserve formants)
            # Only apply 50% of the time to maintain original acoustic data representation
            if np.random.rand() > 0.5:
                n_steps = np.random.randint(-2, 3) 
                if n_steps != 0:
                    # Check if the shifted label remains inside the 88-key piano bounds
                    if 0 <= label + n_steps <= 87:
                        shifted_np = librosa.effects.pitch_shift(waveform.numpy()[0], sr=sample_rate, n_steps=n_steps)
                        waveform = torch.from_numpy(shifted_np).unsqueeze(0)
                        label = label + n_steps

        # Convert to Mel Spectrogram
        if self.transform is None:
            # Default transform
            n_fft = min(1024, self.n_samples) # Ensure n_fft is not larger than signal
            mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=min(256, n_fft//4),
                n_mels=128
            )
            spec = mel_spec_transform(waveform)
        else:
            spec = self.transform(waveform)
            
        # Convert to log scale (decibels) to match human perception
        # Add small value to avoid log(0)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)

        return spec, label

class PreprocessedPianoDataset(Dataset):
    """
    Fast dataset that loads pre-computed mel spectrograms from disk.
    Use preprocess.py to generate the files before training.
    Augmentation here is limited to cheap tensor ops (gain + noise).
    Supports num_workers > 0 safely (no librosa in __getitem__).
    """
    def __init__(self, manifest_path: str, augment: bool = False):
        with open(manifest_path, 'r') as f:
            self.entries = json.load(f)
        self.augment = augment
        print(f"Loaded {len(self.entries)} pre-computed spectrograms from manifest.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        spec = torch.load(entry["path"], weights_only=True)
        label = entry["label"]

        if self.augment:
            # Random gain ±6 dB
            gain = np.random.uniform(0.5, 2.0)
            spec = spec * gain

            # Subtle white noise
            noise_amp = np.random.uniform(0, 0.005)
            spec = spec + torch.randn_like(spec) * noise_amp

        return spec, label


if __name__ == '__main__':
    # Test dataloader
    data_dir = r"C:\Users\ToreGrüner\SkyDrive\Dokumenter\Musikk\Piano"
    dataset = PianoToneDataset(data_dir, augment=True)
    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"Spectrogram shape: {spec.shape}, Label: {label}")
