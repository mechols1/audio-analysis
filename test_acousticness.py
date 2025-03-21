import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import json
from tabulate import tabulate
from custom_audio_analysis import CustomAudioAnalysis

def test_acousticness(audio_dir, output_dir="test_results"):
    """
    Test the improved acousticness calculation on a directory of audio files.
    
    Parameters:
    audio_dir (str): Directory containing audio files for testing
    output_dir (str): Directory to save test results
    
    Returns:
    dict: Results of the acousticness test
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all audio files in the directory
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            audio_files.append(os.path.join(audio_dir, file))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return {}
    
    print(f"Found {len(audio_files)} audio files for testing")
    
    # Dictionary to store results
    results = {}
    
    # Table data for display
    table_data = []
    headers = ["Filename", "Original Acousticness", "Improved Acousticness", "Diff"]
    
    # Process each audio file
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        print(f"Processing {filename}...")
        
        # Create analyzer instance
        analyzer = CustomAudioAnalysis(file_path=audio_file)
        
        # Load audio
        analyzer.load_audio()
        
        # Calculate original acousticness (simple spectral centroid method)
        spectral_centroid = librosa.feature.spectral_centroid(y=analyzer.y, sr=analyzer.sr).mean()
        original_acousticness = float(1.0 - (spectral_centroid / (analyzer.sr/2)) * 0.5)
        original_acousticness = min(max(original_acousticness, 0), 1)
        
        # Calculate improved acousticness
        improved_acousticness = analyzer.calculate_improved_acousticness()
        
        # Store results
        results[filename] = {
            "original_acousticness": original_acousticness,
            "improved_acousticness": improved_acousticness,
            "difference": improved_acousticness - original_acousticness
        }
        
        # Add to table data
        table_data.append([
            filename, 
            f"{original_acousticness:.4f}", 
            f"{improved_acousticness:.4f}",
            f"{improved_acousticness - original_acousticness:+.4f}"
        ])
    
    # Sort table by improved acousticness score (descending)
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    # Print results table
    print("\nAcousticness Test Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Export results to JSON
    with open(os.path.join(output_dir, "acousticness_test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_comparison_plot(results, output_dir, audio_dir)
    
    return results

def create_comparison_plot(results, output_dir, audio_dir):
    """Create a bar chart comparing original vs improved acousticness"""
    filenames = list(results.keys())
    # Sort by improved acousticness for better visualization
    filenames.sort(key=lambda x: results[x]["improved_acousticness"], reverse=True)
    
    original_scores = [results[f]["original_acousticness"] for f in filenames]
    improved_scores = [results[f]["improved_acousticness"] for f in filenames]
    
    # Shorten filenames for display
    display_names = [os.path.splitext(f)[0][:15] + "..." if len(os.path.splitext(f)[0]) > 15 
                    else os.path.splitext(f)[0] for f in filenames]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(filenames))
    width = 0.35
    
    plt.bar(x - width/2, original_scores, width, label='Original Acousticness')
    plt.bar(x + width/2, improved_scores, width, label='Improved Acousticness')
    
    plt.ylabel('Acousticness Score')
    plt.title('Comparison of Acousticness Calculation Methods')
    plt.xticks(x, display_names, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "acousticness_comparison.png"))
    print(f"Comparison plot saved to {os.path.join(output_dir, 'acousticness_comparison.png')}")
    plt.close()
    
    # Create a detailed feature breakdown for the most divergent sample
    max_diff_file = max(results.items(), key=lambda x: abs(x[1]["difference"]))[0]
    max_diff_path = os.path.join(audio_dir, max_diff_file)
    create_feature_breakdown(max_diff_path, output_dir)

def create_feature_breakdown(audio_file, output_dir):
    """Create a detailed breakdown of acousticness features for a specific file"""
    analyzer = CustomAudioAnalysis(file_path=audio_file)
    analyzer.load_audio()
    
    # Extract all the individual components for the improved acousticness
    n_fft = 2048
    hop_length = 512
    
    # Calculate all the individual features
    spectral_centroid = librosa.feature.spectral_centroid(y=analyzer.y, sr=analyzer.sr, 
                                                    n_fft=n_fft, hop_length=hop_length).mean()
    centroid_factor = 1.0 - min((spectral_centroid / (analyzer.sr/2)) * 0.7, 0.7)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=analyzer.y, 
                                                    n_fft=n_fft, hop_length=hop_length).mean()
    flatness_factor = 1.0 - min(spectral_flatness * 10, 1.0)
    
    y_harmonic, y_percussive = librosa.effects.hpss(analyzer.y)
    harmonic_energy = np.sum(y_harmonic**2)
    total_energy = np.sum(analyzer.y**2) if np.sum(analyzer.y**2) > 0 else 1
    harmonic_ratio = harmonic_energy / total_energy
    
    zcr = librosa.feature.zero_crossing_rate(y=analyzer.y, hop_length=hop_length).mean()
    zcr_factor = 1.0 - min(zcr * 5, 1.0)
    
    contrast = librosa.feature.spectral_contrast(y=analyzer.y, sr=analyzer.sr, 
                                            n_fft=n_fft, hop_length=hop_length)
    mean_contrast = np.mean(contrast)
    contrast_factor = min(mean_contrast / 20, 1.0)
    
    bandwidth = librosa.feature.spectral_bandwidth(y=analyzer.y, sr=analyzer.sr, 
                                            n_fft=n_fft, hop_length=hop_length).mean()
    bandwidth_factor = 1.0 - min((bandwidth / (analyzer.sr/2)), 1.0)
    
    S = np.abs(librosa.stft(analyzer.y, n_fft=n_fft, hop_length=hop_length))
    if S.shape[1] > 1:
        harmonic_var = np.var(np.sum(S, axis=0))
        harmonic_var_factor = min(harmonic_var / np.mean(S)**2, 1.0)
    else:
        harmonic_var_factor = 0.5
    
    # Create feature breakdown visualization
    features = {
        "Centroid Factor (25%)": centroid_factor * 0.25,
        "Flatness Factor (20%)": flatness_factor * 0.20,
        "Harmonic Ratio (15%)": harmonic_ratio * 0.15,
        "Contrast Factor (15%)": contrast_factor * 0.15,
        "ZCR Factor (10%)": zcr_factor * 0.10,
        "Bandwidth Factor (10%)": bandwidth_factor * 0.10,
        "Harmonic Var (5%)": harmonic_var_factor * 0.05
    }
    
    # Calculate total acousticness
    improved_acousticness = sum(features.values())
    
    # Calculate original acousticness
    original_acousticness = float(1.0 - (spectral_centroid / (analyzer.sr/2)) * 0.5)
    original_acousticness = min(max(original_acousticness, 0), 1)
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    names = list(features.keys())
    values = list(features.values())
    
    # Sort by contribution (largest first)
    names, values = zip(*sorted(zip(names, values), key=lambda x: x[1], reverse=True))
    
    plt.barh(names, values)
    
    # Add a vertical line for original acousticness
    plt.axvline(x=original_acousticness, color='r', linestyle='--', label=f'Original: {original_acousticness:.4f}')
    
    # Add text for improved acousticness
    plt.text(improved_acousticness - 0.05, -0.5, f'Improved: {improved_acousticness:.4f}', 
             color='blue', fontweight='bold')
    
    plt.xlabel('Contribution to Acousticness Score')
    plt.title(f'Feature Breakdown for {os.path.basename(audio_file)}')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "feature_breakdown.png"))
    print(f"Feature breakdown saved to {os.path.join(output_dir, 'feature_breakdown.png')}")
    plt.close()

if __name__ == "__main__":
    # Check if a directory was provided as command line argument
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
    else:
        # Default test directory
        audio_dir = "test_audio_samples"
        print(f"No audio directory specified. Using default: {audio_dir}")
        
        # Create default directory if it doesn't exist
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            print(f"Created directory: {audio_dir}")
            print("Please add audio files to this directory and run the test again.")
            sys.exit(0)
    
    # Run the test
    test_acousticness(audio_dir)