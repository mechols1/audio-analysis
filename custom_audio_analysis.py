import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pydub import AudioSegment
import requests
from io import BytesIO

class CustomAudioAnalysis:
    def __init__(self, file_path=None, url=None):
        """Initialize with either a local file path or a URL to an audio file"""
        self.file_path = file_path
        self.url = url
        self.y = None
        self.sr = None
        self.duration = None
        self.features = {}
        
    def load_audio(self):
        """Load audio file from either local path or URL"""
        try:
            if self.file_path and os.path.exists(self.file_path):
                self.y, self.sr = librosa.load(self.file_path, sr=None)
                print(f"Audio loaded from file: {self.file_path}")
            elif self.url:
                # Download file from URL
                response = requests.get(self.url)
                audio = BytesIO(response.content)
                # Convert to wav for librosa compatibility
                audio_segment = AudioSegment.from_file(audio)
                audio_segment.export("temp_audio.wav", format="wav")
                self.y, self.sr = librosa.load("temp_audio.wav", sr=None)
                os.remove("temp_audio.wav")  # Clean up temp file
                print(f"Audio loaded from URL: {self.url}")
            else:
                raise ValueError("No valid file path or URL provided")
                
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            print(f"Audio duration: {self.duration:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
            
    def analyze(self):
        """Perform comprehensive audio analysis"""
        if self.y is None:
            success = self.load_audio()
            if not success:
                return None
                
        # Calculate basic features
        self._analyze_basic_features()
        
        # Calculate sections and beats
        self._analyze_sections()
        
        # Calculate segment-level features
        self._analyze_segments()
        
        return self.features
        
    def _detect_key_advanced(self, y=None, sr=None):
        """
        Advanced key detection using multiple approaches and confidence weighting.
        
        Parameters:
        y (numpy.ndarray): Audio time series, if None uses self.y
        sr (int): Sample rate, if None uses self.sr
        
        Returns:
        tuple: (key_number, key_name, mode, mode_name, confidence)
        """
        if y is None:
            y = self.y
        if sr is None:
            sr = self.sr
            
        # Define more accurate Krumhansl-Schmuckler key profiles
        # These are empirically derived weightings of the 12 chromatic pitch classes
        key_profiles = {
            'major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
            'minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        }
        
        # Normalize the key profiles
        for key in key_profiles:
            key_profiles[key] = key_profiles[key] / np.sum(key_profiles[key])
        
        # Extract a higher resolution chromagram with better frequency resolution
        chroma_cqt = librosa.feature.chroma_cqt(
            y=y, 
            sr=sr,
            hop_length=512,
            n_chroma=12, 
            fmin=librosa.note_to_hz('C2'),
            bins_per_octave=36  # Higher resolution
        )
        
        # Also compute HPCP (Harmonic Pitch Class Profile) which can be more robust
        # First get the harmonic content by using a harmonic-percussive source separation
        y_harmonic = librosa.effects.harmonic(y=y, margin=4.0)
        
        # Get chromagram from harmonic component with energy normalization
        chroma_harm = librosa.feature.chroma_cens(
            y=y_harmonic, 
            sr=sr,
            hop_length=512,
            n_chroma=12,
            bins_per_octave=36
        )
        
        # Also compute STFT-based chromagram for another perspective
        chroma_stft = librosa.feature.chroma_stft(
            y=y, 
            sr=sr,
            hop_length=512,
            n_chroma=12
        )
        
        # Combine the three chromagram representations with weights
        chroma_combined = (chroma_cqt * 0.5) + (chroma_harm * 0.33) + (chroma_stft * 0.17)
        
        # For longer tracks, apply exponential weighting to emphasize more recent parts
        if self.duration > 60:  # For tracks longer than 1 minute
            weights = np.exp(np.linspace(-2, 0, chroma_combined.shape[1]))
            chroma_weighted = chroma_combined * weights
            chroma_combined = chroma_weighted
        
        # Compute average chroma vector over time
        chroma_avg = np.mean(chroma_combined, axis=1)
        
        # Compute the correlation of the chroma vector with each key profile
        # for all possible rotations (representing the 12 different keys)
        key_scores = {
            'major': np.zeros(12),
            'minor': np.zeros(12)
        }
        
        # Calculate key scores for all keys
        for mode in ['major', 'minor']:
            profile = key_profiles[mode]
            for key_idx in range(12):
                # Rotate profile to represent different keys
                rotated_profile = np.roll(profile, key_idx)
                # Calculate correlation
                corr = np.corrcoef(chroma_avg, rotated_profile)[0, 1]
                key_scores[mode][key_idx] = corr if not np.isnan(corr) else -1.0
        
        # Find key with highest correlation
        max_major_key = np.argmax(key_scores['major'])
        max_minor_key = np.argmax(key_scores['minor'])
        
        max_major_val = key_scores['major'][max_major_key]
        max_minor_val = key_scores['minor'][max_minor_key]
        
        # Key map for converting numeric key to name
        key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Determine overall key (major or minor) by highest correlation
        if max_major_val > max_minor_val:
            key = max_major_key
            key_name = key_map[key]
            mode = 1  # 1 for major
            mode_name = "Major"
            confidence = max_major_val
        else:
            key = max_minor_key
            key_name = key_map[key]
            mode = 0  # 0 for minor
            mode_name = "Minor"
            confidence = max_minor_val
        
        # Enhance analysis with segment-based voting for more robust results
        # Divide the track into segments and analyze each segment separately
        segment_length = int(sr * 5)  # 5-second segments
        num_segments = len(y) // segment_length
        
        # Skip segment voting for very short tracks
        if num_segments > 1:
            segment_keys = []
            segment_confidences = []
            
            for i in range(num_segments):
                start = i * segment_length
                end = min((i + 1) * segment_length, len(y))
                segment = y[start:end]
                
                # Skip segments with low energy (likely silence or noise)
                if np.mean(segment**2) < 0.001:
                    continue
                    
                # Perform key detection on this segment
                chroma_segment = librosa.feature.chroma_cqt(y=segment, sr=sr)
                chroma_avg_segment = np.mean(chroma_segment, axis=1)
                
                # Calculate scores for this segment
                segment_key_scores = {
                    'major': np.zeros(12),
                    'minor': np.zeros(12)
                }
                
                for mode in ['major', 'minor']:
                    profile = key_profiles[mode]
                    for key_idx in range(12):
                        rotated_profile = np.roll(profile, key_idx)
                        corr = np.corrcoef(chroma_avg_segment, rotated_profile)[0, 1]
                        segment_key_scores[mode][key_idx] = corr if not np.isnan(corr) else -1.0
                
                # Determine key for this segment
                seg_max_major_key = np.argmax(segment_key_scores['major'])
                seg_max_minor_key = np.argmax(segment_key_scores['minor'])
                
                seg_max_major_val = segment_key_scores['major'][seg_max_major_key]
                seg_max_minor_val = segment_key_scores['minor'][seg_max_minor_key]
                
                if seg_max_major_val > seg_max_minor_val:
                    segment_keys.append((seg_max_major_key, 1))  # (key, mode)
                    segment_confidences.append(seg_max_major_val)
                else:
                    segment_keys.append((seg_max_minor_key, 0))  # (key, mode)
                    segment_confidences.append(seg_max_minor_val)
            
            # Weight segment votes by their confidence
            if segment_keys:
                # Count occurrences of each key-mode combination
                key_mode_counts = {}
                for i, (seg_key, seg_mode) in enumerate(segment_keys):
                    key_mode = (seg_key, seg_mode)
                    if key_mode not in key_mode_counts:
                        key_mode_counts[key_mode] = 0
                    key_mode_counts[key_mode] += segment_confidences[i]
                
                # Find the most common key-mode combination
                if key_mode_counts:
                    best_key_mode = max(key_mode_counts.items(), key=lambda x: x[1])[0]
                    best_seg_key, best_seg_mode = best_key_mode
                    
                    # If the segment-based key has high confidence, use it instead
                    segment_weight = 0.4  # How much we trust segment analysis vs. global
                    segment_confidence = key_mode_counts[best_key_mode] / sum(segment_confidences)
                    
                    if segment_confidence > 0.3:  # Threshold for considering segment analysis
                        # Blend the global and segment analysis
                        if best_seg_mode == mode:  # If modes match
                            # Still might be different keys in same mode
                            if best_seg_key != key:
                                # Check if they're harmonically related (circle of fifths)
                                fifth_distance = min((key - best_seg_key) % 12, (best_seg_key - key) % 12)
                                if fifth_distance <= 1:  # Adjacent on circle of fifths
                                    # Average the confidences
                                    confidence = (confidence + segment_confidence) / 2
                                elif segment_confidence > confidence + 0.1:
                                    # If segment confidence is significantly higher, use segment key
                                    key = best_seg_key
                                    key_name = key_map[key]
                                    confidence = segment_confidence
                        elif segment_confidence > confidence + 0.15:
                            # If segment mode differs but has much higher confidence, use it
                            key = best_seg_key
                            key_name = key_map[key]
                            mode = best_seg_mode
                            mode_name = "Major" if mode == 1 else "Minor"
                            confidence = segment_confidence
        
        # Additional consideration for relative major/minor
        # Sometimes algorithms confuse relative keys (e.g., C major vs. A minor)
        # Check relative major/minor correlation
        relative_minor_idx = (key + 9) % 12 if mode == 1 else key
        relative_major_idx = (key + 3) % 12 if mode == 0 else key
        
        relative_minor_val = key_scores['minor'][relative_minor_idx]
        relative_major_val = key_scores['major'][relative_major_idx]
        
        # If relative minor/major has very close score, consider ambiguity
        if mode == 1 and abs(max_major_val - relative_minor_val) < 0.05:
            # Very ambiguous between major and relative minor
            # Consider additional features to disambiguate
            
            # Check for presence of strong V-I cadences (major indicator)
            # Simple heuristic: look for strong dominant (V) to tonic (I) transitions
            chroma_product = np.roll(chroma_avg, -7) * chroma_avg  # Correlate V with I
            major_cadence_strength = np.sum(chroma_product)
            
            # Check for presence of minor third interval (minor indicator)
            minor_third_strength = chroma_avg[(key + 3) % 12]
            major_third_strength = chroma_avg[(key + 4) % 12]
            
            if minor_third_strength > major_third_strength + 0.1 and major_cadence_strength < 0.15:
                # Switch to minor if minor third is much stronger
                key = relative_minor_idx
                key_name = key_map[key]
                mode = 0
                mode_name = "Minor"
                confidence = relative_minor_val
        
        elif mode == 0 and abs(max_minor_val - relative_major_val) < 0.05:
            # Similar check for minor keys that might actually be major
            major_third_strength = chroma_avg[(key + 4) % 12]
            minor_third_strength = chroma_avg[(key + 3) % 12]
            
            if major_third_strength > minor_third_strength + 0.1:
                # Switch to major if major third is much stronger
                key = relative_major_idx
                key_name = key_map[key]
                mode = 1
                mode_name = "Major"
                confidence = relative_major_val
        
        return key, key_name, mode, mode_name, confidence

    def calculate_improved_danceability(self, y=None, sr=None):
        """
        Calculate improved danceability based on multiple audio features.
        
        Danceability represents how suitable a track is for dancing based on
        a combination of musical elements including rhythm stability, beat strength,
        tempo, and rhythmic pattern regularity.
        
        Parameters:
        y (numpy.ndarray): Audio time series, if None uses self.y
        sr (int): Sample rate, if None uses self.sr
        
        Returns:
        float: Danceability score between 0.0 and 1.0
        """
        if y is None:
            y = self.y
        if sr is None:
            sr = self.sr
        
        # 1. Calculate onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        
        # 2. Beat detection and strength
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Calculate beat confidence using beat_plp
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        beat_strength = np.mean(pulse)
        
        # 3. Calculate tempogram for rhythm analysis
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        
        # Measure beat regularity using tempogram statistics
        # Lower standard deviation means more regular beats
        tempo_std = np.std(np.std(tempogram, axis=1))
        beat_regularity = 1.0 - min(tempo_std / 2.0, 0.5)  # Normalize to 0-0.5 range
        
        # 4. Split the signal into harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate low-frequency energy ratio (bass presence)
        n_fft = 2048
        hop_length = 512
        
        # Get the spectrogram
        S_full = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        # Define the low-frequency range (approximately up to 250 Hz)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        low_freq_bins = (freq_bins <= 250)
        
        # Calculate the ratio of low-frequency energy to total energy
        low_energy = np.sum(S_full[low_freq_bins, :])
        total_energy = np.sum(S_full)
        bass_presence = low_energy / total_energy if total_energy > 0 else 0
        
        # 5. Tempo factor - certain tempo ranges are more danceable
        # Most danceable range is around 90-130 BPM
        tempo_factor = 0.0
        if 90 <= tempo <= 130:
            # Ideal dance tempo range
            tempo_factor = 1.0
        elif 75 <= tempo < 90 or 130 < tempo <= 150:
            # Good but not ideal range
            tempo_factor = 0.8
        elif 60 <= tempo < 75 or 150 < tempo <= 180:
            # Less optimal but still danceable
            tempo_factor = 0.6
        else:
            # Outside typical dance tempo range
            tempo_factor = 0.4
        
        # 6. Calculate onset density (number of onsets per second)
        # More onsets can indicate more rhythmic activity
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(y) > 0:
            duration = len(y) / sr
            if duration > 0:
                onset_density = len(onset_frames) / duration
                # Normalize onset density (typical range 0.5-5 onsets per second)
                onset_density_norm = min(onset_density / 5.0, 1.0)
            else:
                onset_density_norm = 0.5
        else:
            onset_density_norm = 0.5
        
        # 7. Percussive energy ratio
        if np.sum(y**2) > 0:
            percussive_ratio = np.sum(y_percussive**2) / np.sum(y**2)
        else:
            percussive_ratio = 0.5
        
        # Combine all factors with appropriate weights
        danceability = (
            0.30 * beat_strength +
            0.15 * beat_regularity +
            0.15 * tempo_factor +
            0.20 * bass_presence +
            0.10 * percussive_ratio +
            0.10 * onset_density_norm +
            0.05 * (1.0 - tempo_std)  # Rhythm consistency
        )
        
        # Additional boost for tracks with clear rhythmic patterns
        # and appropriate tempo for dancing
        if beat_strength > 0.6 and 90 <= tempo <= 130 and beat_regularity > 0.7:
            danceability = min(danceability * 1.2, 1.0)
        
        # Scale to ensure a good distribution between 0 and 1
        danceability = min(max(danceability, 0.0), 1.0)
        
        return float(danceability)

    def calculate_improved_energy(self, y=None, sr=None):
        """
        Calculate improved energy metric that better captures perceived energy in music.
        
        Energy represents the perceived intensity and activity in an audio track,
        combining multiple factors including dynamics, spectral characteristics,
        and rhythmic intensity.
        
        Parameters:
        y (numpy.ndarray): Audio time series, if None uses self.y
        sr (int): Sample rate, if None uses self.sr
        
        Returns:
        float: Energy score between 0.0 and 1.0
        """
        if y is None:
            y = self.y
        if sr is None:
            sr = self.sr
        
        # 1. Root Mean Square energy (raw dynamics)
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Normalize RMS (typical values are between 0.02-0.2)
        rms_norm = min(rms_mean / 0.2, 1.0)
        
        # 2. Calculate spectral energy distribution
        # High energy songs have more energy in higher frequencies
        S = np.abs(librosa.stft(y))
        
        # Calculate spectral centroid (center of mass of the spectrum)
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroid)
        
        # Normalize spectral centroid
        # Higher centroid = more high frequency content = more energy
        centroid_norm = min(centroid_mean / (sr/4), 1.0)
        
        # 3. Calculate spectral contrast
        # High contrast indicates more dynamic range in frequency bands
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        contrast_mean = np.mean(np.mean(contrast, axis=1))
        
        # Normalize contrast
        contrast_norm = min(contrast_mean / 50, 1.0)
        
        # 4. Calculate spectral flatness
        # Less flat (more tonal) often means more energy
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = np.mean(flatness)
        
        # Invert flatness: lower flatness often means more tonal clarity and energy
        tonal_norm = 1.0 - min(flatness_mean * 10, 1.0)
        
        # 5. Calculate temporal dynamics (variance in loudness)
        # Dynamic tracks often feel more energetic
        dynamics_norm = min(rms_std / 0.1, 1.0)
        
        # 6. Get percussive content
        # Separate harmonic and percussive elements
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate percussive energy ratio
        if np.sum(y**2) > 0:
            percussive_ratio = np.sum(y_percussive**2) / np.sum(y**2)
        else:
            percussive_ratio = 0.5
        
        # 7. Onset strength (indicates rhythmic activity)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env)
        
        # Normalize onset strength
        onset_norm = min(onset_mean / 0.5, 1.0)
        
        # 8. Tempo factor
        # Faster tempos often correspond to higher energy
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Normalize tempo (most energetic range is above 120 BPM)
        if tempo < 70:
            tempo_factor = 0.4
        elif 70 <= tempo < 110:
            tempo_factor = 0.6
        elif 110 <= tempo < 140:
            tempo_factor = 0.8
        else:  # tempo >= 140
            tempo_factor = 1.0
        
        # 9. Calculate spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_mean = np.mean(rolloff)
        
        # Normalize rolloff
        rolloff_norm = min(rolloff_mean / (sr/2), 1.0)
        
        # Combine all factors with weights to get final energy value
        energy = (
            0.20 * rms_norm +           # Raw loudness
            0.15 * centroid_norm +      # High-frequency content
            0.10 * contrast_norm +      # Spectral contrast
            0.05 * tonal_norm +         # Tonal clarity
            0.05 * dynamics_norm +      # Dynamic range
            0.15 * percussive_ratio +   # Percussive content
            0.15 * onset_norm +         # Rhythmic activity
            0.10 * tempo_factor +       # Tempo influence
            0.05 * rolloff_norm         # Spectral rolloff
        )
        
        # Make sure the result is between 0 and 1
        energy = min(max(energy, 0.0), 1.0)
        
        return float(energy)

    def calculate_improved_acousticness(self, y=None, sr=None):
        """
        Calculate an improved acousticness measure based on multiple audio features.
        
        Acousticness represents the confidence that a track contains primarily acoustic 
        instruments (vs. electronic/synthetic sounds). This implementation uses multiple 
        spectral and temporal features for a more robust estimation.
        
        Parameters:
        y (numpy.ndarray): Audio time series, if None uses self.y
        sr (int): Sample rate, if None uses self.sr
        
        Returns:
        float: Acousticness score between 0.0 and 1.0
        """
        if y is None:
            y = self.y
        if sr is None:
            sr = self.sr
        
        n_fft = 2048
        hop_length = 512
        
        # 1. Spectral centroid - lower values typically indicate acoustic instruments
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                            n_fft=n_fft, 
                                                            hop_length=hop_length).mean()
        # Normalize by Nyquist frequency and invert (lower centroids = higher acousticness)
        centroid_factor = 1.0 - min((spectral_centroid / (sr/2)) * 0.7, 0.7)
        
        # 2. Spectral flatness - acoustic sounds tend to have less flat spectra
        # (more peaks and valleys in the spectrum)
        spectral_flatness = librosa.feature.spectral_flatness(y=y, 
                                                            n_fft=n_fft, 
                                                            hop_length=hop_length).mean()
        # Lower flatness = higher acousticness (invert and scale)
        flatness_factor = 1.0 - min(spectral_flatness * 10, 1.0)
        
        # 3. Harmonic vs percussive decomposition
        # Acoustic instruments often have stronger harmonic content
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate ratio of harmonic energy to total energy
        harmonic_energy = np.sum(y_harmonic**2)
        total_energy = np.sum(y**2) if np.sum(y**2) > 0 else 1
        harmonic_ratio = harmonic_energy / total_energy
        
        # 4. Zero-crossing rate - acoustic sounds often have lower ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y, 
                                            hop_length=hop_length).mean()
        # Lower ZCR = higher acousticness (invert and scale)
        zcr_factor = 1.0 - min(zcr * 5, 1.0)  # Scale factor may need adjustment
        
        # 5. Spectral contrast - acoustic instruments often have higher contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length)
        # Take mean of contrast across all frequency bands
        mean_contrast = np.mean(contrast)
        # Scale contrast to 0-1 range (higher contrast = higher acousticness)
        contrast_factor = min(mean_contrast / 20, 1.0)  # Scale factor may need adjustment
        
        # 6. Spectral bandwidth - acoustic sounds often have more focused bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                    n_fft=n_fft, 
                                                    hop_length=hop_length).mean()
        # Normalize by Nyquist frequency and invert
        bandwidth_factor = 1.0 - min((bandwidth / (sr/2)), 1.0)
        
        # 7. Harmonic ratio variance - acoustic instruments often have more 
        # variation in harmonic content over time
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        if S.shape[1] > 1:  # Ensure we have enough frames
            harmonic_var = np.var(np.sum(S, axis=0))
            # Normalize variance (higher variance = higher acousticness)
            harmonic_var_factor = min(harmonic_var / np.mean(S)**2, 1.0)
        else:
            harmonic_var_factor = 0.5  # Default if not enough frames
        
        # Combine all factors with appropriate weights
        acousticness = (
            0.25 * centroid_factor +      # Most important factor
            0.20 * flatness_factor +      # Strong indicator 
            0.15 * harmonic_ratio +       # Important for harmonic instruments
            0.15 * contrast_factor +      # Important for distinguishing timbres
            0.10 * zcr_factor +           # Helps identify noisy components
            0.10 * bandwidth_factor +     # Helps with spectral focus
            0.05 * harmonic_var_factor    # Adds sensitivity to temporal variations
        )
        
        # Additional adjustments for edge cases:
        # 1. Strongly harmonic with low flatness is very likely acoustic
        if harmonic_ratio > 0.8 and flatness_factor > 0.8:
            acousticness = min(acousticness * 1.2, 1.0)
        
        # 2. Very flat spectrum with high ZCR is very likely electronic
        if flatness_factor < 0.3 and zcr_factor < 0.3:
            acousticness = acousticness * 0.8
            
        # Final normalization
        acousticness = min(max(acousticness, 0.0), 1.0)
        
        return float(acousticness)

    # Replace the key detection part in _analyze_basic_features with this:
    def _analyze_basic_features(self):
        """Extract basic audio features with improved key detection"""
        # [Keep all your existing code before key detection]
        
        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        
        # Improved key detection
        key, key_name, mode, mode_name, key_confidence = self._detect_key_advanced()
        
        # [Continue with your existing code for other features]
        # Loudness
        loudness = librosa.feature.rms(y=self.y).mean()
        
        # Energy
        energy = self.calculate_improved_energy()
        
        # Time signature estimation - improved version
        _, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        if len(beats) > 0:
            # Calculate beat intervals
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=self.sr))
            
            # Use autocorrelation to find the meter
            if len(beat_intervals) > 4:
                # Calculate autocorrelation of beat intervals
                autocorr = librosa.autocorrelate(beat_intervals, max_size=8)
                # Find peaks in autocorrelation
                peaks = librosa.util.peak_pick(autocorr, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=1)
                
                if len(peaks) > 0:
                    # Find the first significant peak (excluding the zero-lag peak)
                    # This peak indicates the most likely meter
                    significant_peaks = peaks[peaks > 0]
                    if len(significant_peaks) > 0:
                        first_peak = significant_peaks[0]
                        # The peak position (+1) gives us a good estimate of the meter
                        time_signature = first_peak + 1
                        time_signature = max(2, min(time_signature, 7))  # Constrain to common time signatures
                    else:
                        time_signature = 4  # Default to 4/4 if no peaks found
                else:
                    time_signature = 4  # Default to 4/4 if no peaks found
            else:
                time_signature = 4  # Default to 4/4 if not enough beats
        else:
            time_signature = 4  # Default to 4/4 if no beats detected
        
        # Improved danceability calculation
        danceability = self.calculate_improved_danceability()
        
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        if np.sum(self.y**2) > 0:
            percussive_ratio = np.sum(y_percussive**2) / np.sum(self.y**2)
        else:
            percussive_ratio = 0.5
        
        # Improved acousticness calculation
        # Use multiple spectral features for better acousticness estimation
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=self.y).mean()
        
        # Acoustic sounds typically have lower spectral centroids and more tonal content
        # Electronic sounds often have higher centroid and more noise-like spectrum (higher flatness)
        normalized_centroid = spectral_centroid / (self.sr/2)  # Normalize to 0-1
        
        # Calculate acousticness using acousticness calculation function
        acousticness = self.calculate_improved_acousticness()
        
        # Improved valence (emotional positivity) calculation
        # Combine multiple factors that contribute to perceived valence
        
        # Spectral contrast captures tonal vs. noise content across frequency bands
        contrast_mean = np.mean(librosa.feature.spectral_contrast(y=self.y, sr=self.sr))
        
        # Major keys typically have higher valence than minor keys
        key_factor = 0.2 if mode == 1 else -0.1  # Bonus for major keys
        
        # Tempo factor - faster tempos often correlate with higher valence
        tempo_factor = min((tempo - 70) / 100, 0.3) if tempo > 70 else -0.1
        
        # Calculate brightness using spectral features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr).mean()
        brightness = spectral_rolloff / (self.sr/2)  # Normalize
        
        # Combine factors for valence
        valence = 0.5 + (contrast_mean / 50) + key_factor + tempo_factor + (brightness * 0.1)
        
        # Constrain to 0-1 range
        valence = min(max(valence, 0), 1)

        # Store in features dictionary
        # Add instrumentalness calculation if it hasn't been calculated yet
        if 'instrumentalness' not in locals():
            # Improved instrumentalness detection (likelihood of no vocals)
            # Using both spectral flatness and MFCCs to detect vocal presence
            mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
            mfcc_var = np.var(mfccs, axis=1).mean()  # Vocal variation in MFCCs
            
            # High MFCC variance often indicates vocal presence
            spec_flatness = librosa.feature.spectral_flatness(y=self.y).mean()
            
            # Combine factors (high flatness + low MFCC variance = more instrumental)
            instrumentalness = spec_flatness * 8 * (1 - min(mfcc_var / 5, 0.8))
            instrumentalness = min(max(instrumentalness, 0), 1)
        
        self.features["track"] = {
            "duration": self.duration,
            "tempo": float(tempo),
            "key": int(key),
            "key_name": key_name,
            "key_confidence": float(key_confidence),
            "mode": mode_name,
            "time_signature": int(time_signature),
            "loudness": float(loudness),
            "energy": float(energy),
            "danceability": float(danceability),
            "acousticness": float(acousticness),
            "instrumentalness": float(instrumentalness),
            "valence": float(valence)
        }
            
    def _analyze_sections(self):
        """Identify structural sections in the track and beat positions"""
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Structural segmentation using spectral clustering
        bounds = librosa.segment.agglomerative(librosa.feature.chroma_cqt(y=self.y, sr=self.sr), 
                                              k=5)  # Assume approximately 5 sections
        bound_times = librosa.frames_to_time(bounds, sr=self.sr)
        
        # Determine section types (simple heuristic)
        section_types = ["intro", "verse", "chorus", "bridge", "outro"]
        sections = []
        
        for i in range(len(bound_times)-1):
            start = bound_times[i]
            end = bound_times[i+1]
            
            # Get audio segment for this section
            start_idx = int(start * self.sr)
            end_idx = min(int(end * self.sr), len(self.y))
            segment = self.y[start_idx:end_idx]
            
            # Calculate features to determine section type
            rms = librosa.feature.rms(y=segment).mean()
            spec_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr).mean()
            
            # Simple heuristic to guess section type
            if i == 0:
                section_type = "intro"
            elif i == len(bound_times)-2:
                section_type = "outro"
            elif rms > np.mean(librosa.feature.rms(y=self.y)):
                section_type = "chorus"
            elif spec_centroid > np.mean(librosa.feature.spectral_centroid(y=self.y, sr=self.sr)):
                section_type = "bridge"
            else:
                section_type = "verse"
                
            # Store section info
            sections.append({
                "start": float(start),
                "duration": float(end - start),
                "loudness": float(rms),
                "type": section_type
            })
            
        # Store beats and sections in features
        self.features["beats"] = [float(time) for time in beat_times]
        self.features["sections"] = sections
        
    def _analyze_segments(self):
        """Analyze audio at segment level (similar to Spotify's segments)"""
        # Frame-level analysis
        frame_length = 2048
        hop_length = 512
        
        # Segment the audio into frames
        segments = []
        frames = range(0, len(self.y), hop_length)
        
        for i, frame_idx in enumerate(frames[:-1]):
            # Skip some frames for efficiency in this example
            if i % 10 != 0 and i < len(frames) - 20:
                continue
                
            # Time information
            start_time = frame_idx / self.sr
            end_time = min((frame_idx + hop_length) / self.sr, self.duration)
            
            # Get segment audio
            segment_y = self.y[frame_idx:frame_idx+frame_length]
            if len(segment_y) < frame_length:
                # Pad if needed
                segment_y = np.pad(segment_y, (0, frame_length - len(segment_y)))
                
            # Loudness
            loudness = librosa.feature.rms(y=segment_y).mean()
            
            # Pitch information (chroma)
            if len(segment_y) > 0:
                chroma = librosa.feature.chroma_cqt(y=segment_y, sr=self.sr)
                pitches = list(np.mean(chroma, axis=1))
            else:
                pitches = [0] * 12
                
            # Timbre (using MFCCs as a simplified representation)
            if len(segment_y) > 0:
                mfccs = librosa.feature.mfcc(y=segment_y, sr=self.sr, n_mfcc=12)
                timbre = list(np.mean(mfccs, axis=1))
            else:
                timbre = [0] * 12
                
            # Store segment data
            segments.append({
                "start": float(start_time),
                "duration": float(end_time - start_time),
                "loudness": float(loudness),
                "pitches": [float(p) for p in pitches],
                "timbre": [float(t) for t in timbre]
            })
            
        self.features["segments"] = segments
        
    def visualize(self, output_dir="./", prefix="track"):
        """Generate visualizations similar to Spotify's audio analysis"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Waveform and spectrogram
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(self.y, sr=self.sr)
        plt.title('Waveform')
        
        plt.subplot(3, 1, 2)
        spectrogram = librosa.amplitude_to_db(
            np.abs(librosa.stft(self.y)), ref=np.max)
        librosa.display.specshow(spectrogram, sr=self.sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Add sections and beats
        plt.subplot(3, 1, 3)
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.5)
        
        # Plot section boundaries
        for section in self.features.get("sections", []):
            plt.axvline(x=section["start"], color='r', linestyle='--', alpha=0.5)
            plt.text(section["start"], 0.5, section["type"], 
                    rotation=90, verticalalignment='center')
            
        # Plot beats
        beat_times = self.features.get("beats", [])
        plt.plot(beat_times, np.zeros_like(beat_times) - 0.1, 'o', color='green', alpha=0.5)
        
        plt.title('Sections and Beats')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"{prefix}_analysis.png"))
        
        # Save features as JSON
        with open(os.path.join(output_dir, f"{prefix}_analysis.json"), 'w') as f:
            json.dump(self.features, f, indent=2)
            
        print(f"Visualizations saved to {output_dir}")
        
        plt.close()
        
    def export_json(self, output_path=None):
        """Export analysis results as JSON"""
        if output_path is None:
            if self.file_path:
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                output_path = f"{base_name}_analysis.json"
            else:
                output_path = "audio_analysis.json"
                
        with open(output_path, 'w') as f:
            json.dump(self.features, f, indent=2)
            
        print(f"Analysis exported to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Example 1: Analyze a local file
    analyzer = CustomAudioAnalysis(file_path="path/to/your/audio_file.mp3")
    analyzer.analyze()
    analyzer.visualize()
    analyzer.export_json()
    
    # Example 2: Analyze from a URL
    # analyzer = CustomAudioAnalysis(url="https://example.com/audio_file.mp3")
    # analyzer.analyze()
    # analyzer.visualize()