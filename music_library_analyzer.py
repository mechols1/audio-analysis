import os
import glob
import json
import numpy as np
from custom_audio_analysis import CustomAudioAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class MusicLibraryAnalyzer:
    def __init__(self, music_dir):
        self.music_dir = music_dir
        self.tracks = []
        self.features_db = {}
        
    def scan_library(self, file_extensions=None):
        """Scan music directory for audio files"""
        if file_extensions is None:
            file_extensions = ['.mp3', '.wav', '.m4a', '.flac']
            
        print(f"Scanning {self.music_dir} for audio files...")
        
        for ext in file_extensions:
            pattern = os.path.join(self.music_dir, f"**/*{ext}")
            self.tracks.extend(glob.glob(pattern, recursive=True))
            
        print(f"Found {len(self.tracks)} tracks")
        return self.tracks
    
    def analyze_library(self, force_reanalyze=False):
        """Analyze all tracks in the library"""
        if not self.tracks:
            self.scan_library()
            
        for track_path in self.tracks:
            # Create output filename for analysis JSON
            track_name = os.path.basename(track_path)
            analysis_path = os.path.join(
                self.music_dir, 
                "analysis", 
                f"{os.path.splitext(track_name)[0]}_analysis.json"
            )
            
            # Skip if analysis already exists
            if os.path.exists(analysis_path) and not force_reanalyze:
                print(f"Loading existing analysis for {track_name}")
                with open(analysis_path, 'r') as f:
                    track_features = json.load(f)
                self.features_db[track_path] = track_features
                continue
                
            # Ensure analysis directory exists
            os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
            
            # Analyze track
            print(f"Analyzing {track_name}...")
            analyzer = CustomAudioAnalysis(file_path=track_path)
            track_features = analyzer.analyze()
            
            if track_features:
                # Save analysis
                analyzer.export_json(analysis_path)
                self.features_db[track_path] = track_features
            else:
                print(f"Failed to analyze {track_name}")
                
        print(f"Analyzed {len(self.features_db)} tracks")
        return self.features_db
    
    def extract_feature_vector(self, track_features):
        """Extract a numerical feature vector from track features"""
        if not track_features:
            return None
            
        # Extract basic audio features
        track_info = track_features.get("track", {})
        
        vector = [
            track_info.get("tempo", 0) / 200,  # Normalize tempo
            track_info.get("key", 0) / 11,  # Normalize key
            track_info.get("mode", 0),  # Binary
            track_info.get("energy", 0),
            track_info.get("danceability", 0),
            track_info.get("acousticness", 0),
            track_info.get("instrumentalness", 0),
            track_info.get("valence", 0),
        ]
        
        return np.array(vector)
    
    def find_similar_tracks(self, reference_track, top_n=5):
        """Find tracks similar to reference track"""
        if not self.features_db:
            print("No analyzed tracks in database. Run analyze_library() first.")
            return []
            
        # Get features of reference track
        if isinstance(reference_track, str) and os.path.exists(reference_track):
            # It's a file path
            if reference_track in self.features_db:
                ref_features = self.features_db[reference_track]
            else:
                # Analyze the track if not in database
                analyzer = CustomAudioAnalysis(file_path=reference_track)
                ref_features = analyzer.analyze()
        else:
            print("Reference track must be a valid file path")
            return []
            
        # Extract feature vector for reference track
        ref_vector = self.extract_feature_vector(ref_features)
        
        if ref_vector is None:
            print("Could not extract features from reference track")
            return []
            
        # Calculate similarity to all tracks
        similarities = []
        
        for track_path, track_features in self.features_db.items():
            if track_path == reference_track:
                continue  # Skip reference track
                
            track_vector = self.extract_feature_vector(track_features)
            
            if track_vector is not None:
                # Calculate cosine similarity
                sim_score = cosine_similarity([ref_vector], [track_vector])[0][0]
                similarities.append((track_path, sim_score))
                
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N similar tracks
        return similarities[:top_n]
    
    def create_playlist(self, seed_tracks, size=10, min_tempo=None, max_tempo=None, target_energy=None):
        """Create a playlist based on seed tracks with optional constraints"""
        if not self.features_db:
            print("No analyzed tracks in database. Run analyze_library() first.")
            return []
            
        # Get feature vectors for seed tracks
        seed_vectors = []
        
        for track in seed_tracks:
            if track in self.features_db:
                seed_vector = self.extract_feature_vector(self.features_db[track])
                if seed_vector is not None:
                    seed_vectors.append(seed_vector)
                    
        if not seed_vectors:
            print("No valid seed tracks found")
            return []
            
        # Calculate average seed vector
        target_vector = np.mean(seed_vectors, axis=0)
        
        # Get all tracks with similarity scores
        scored_tracks = []
        
        for track_path, track_features in self.features_db.items():
            if track_path in seed_tracks:
                continue  # Skip seed tracks
                
            track_vector = self.extract_feature_vector(track_features)
            
            if track_vector is not None:
                # Check constraints
                meets_constraints = True
                
                if min_tempo is not None or max_tempo is not None:
                    tempo = track_features.get("track", {}).get("tempo", 0)
                    
                    if min_tempo is not None and tempo < min_tempo:
                        meets_constraints = False
                        
                    if max_tempo is not None and tempo > max_tempo:
                        meets_constraints = False
                        
                if target_energy is not None:
                    energy = track_features.get("track", {}).get("energy", 0)
                    # Allow some wiggle room
                    if abs(energy - target_energy) > 0.2:
                        meets_constraints = False
                        
                if meets_constraints:
                    # Calculate cosine similarity
                    sim_score = cosine_similarity([target_vector], [track_vector])[0][0]
                    scored_tracks.append((track_path, sim_score))
                    
        # Sort by similarity (highest first)
        scored_tracks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top tracks as playlist
        return scored_tracks[:size]
    
    def visualize_library(self, output_path="library_analysis.png"):
        """Visualize tracks in the library by features"""
        if not self.features_db:
            print("No analyzed tracks in database. Run analyze_library() first.")
            return
            
        # Extract relevant features
        energies = []
        danceabilities = []
        valences = []
        tempos = []
        track_names = []
        
        for track_path, features in self.features_db.items():
            track_info = features.get("track", {})
            
            track_names.append(os.path.basename(track_path))
            energies.append(track_info.get("energy", 0))
            danceabilities.append(track_info.get("danceability", 0))
            valences.append(track_info.get("valence", 0))
            tempos.append(track_info.get("tempo", 0))
            
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Energy vs. Danceability
        plt.subplot(2, 2, 1)
        plt.scatter(danceabilities, energies, alpha=0.6)
        plt.xlabel('Danceability')
        plt.ylabel('Energy')
        plt.title('Energy vs. Danceability')
        plt.grid(True, alpha=0.3)
        
        # Energy vs. Valence
        plt.subplot(2, 2, 2)
        plt.scatter(valences, energies, alpha=0.6)
        plt.xlabel('Valence (Positivity)')
        plt.ylabel('Energy')
        plt.title('Energy vs. Valence')
        plt.grid(True, alpha=0.3)
        
        # Tempo distribution
        plt.subplot(2, 2, 3)
        plt.hist(tempos, bins=20, alpha=0.7)
        plt.xlabel('Tempo (BPM)')
        plt.ylabel('Number of Tracks')
        plt.title('Tempo Distribution')
        plt.grid(True, alpha=0.3)
        
        # Danceability vs. Valence
        plt.subplot(2, 2, 4)
        plt.scatter(valences, danceabilities, alpha=0.6)
        plt.xlabel('Valence (Positivity)')
        plt.ylabel('Danceability')
        plt.title('Danceability vs. Valence')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Library visualization saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize with your music directory
    library = MusicLibraryAnalyzer("path/to/your/music/library")
    
    # Scan and analyze the library
    library.scan_library()
    library.analyze_library()
    
    # Visualize the library
    library.visualize_library()
    
    # Find similar tracks to a reference track
    similar_tracks = library.find_similar_tracks("path/to/your/music/library/favorite_song.mp3")
    print("\nSimilar tracks:")
    for track, similarity in similar_tracks:
        print(f"{os.path.basename(track)}: {similarity:.2f}")
    
    # Create a playlist based on seed tracks
    seed_tracks = [
        "path/to/your/music/library/seed_song1.mp3",
        "path/to/your/music/library/seed_song2.mp3"
    ]
    
    playlist = library.create_playlist(
        seed_tracks=seed_tracks,
        size=10,
        min_tempo=100,
        max_tempo=130,
        target_energy=0.7
    )
    
    print("\nGenerated playlist:")
    for track, similarity in playlist:
        print(f"{os.path.basename(track)}: {similarity:.2f}")