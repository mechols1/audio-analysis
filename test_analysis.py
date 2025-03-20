from custom_audio_analysis import CustomAudioAnalysis
import librosa 
import os

# Path to a test audio file - replace with an actual file on your system
TEST_AUDIO_FILE = "/Users/isrealechols/Downloads/NeverTooMuch.mp3"  # Replace this!

def test_basic_analysis():
    """Test basic audio analysis functionality"""
    print(f"Testing analysis with file: {TEST_AUDIO_FILE}")
    
    # Verify file exists
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"Error: Test file not found at {TEST_AUDIO_FILE}")
        return False
    
    # Create analyzer
    analyzer = CustomAudioAnalysis(file_path=TEST_AUDIO_FILE)
    
    # Run analysis
    print("Starting analysis...")
    features = analyzer.analyze()
    
    # Check if analysis succeeded
    if not features:
        print("Error: Analysis failed")
        return False
    
    # Print basic track info
    track_info = features.get("track", {})
    print("\n--- Basic Track Analysis Results ---")
    print(f"Duration: {track_info.get('duration', 'Unknown'):.2f} seconds")
    print(f"Tempo: {track_info.get('tempo', 'Unknown'):.1f} BPM")
    print(f"Key: {track_info.get('key_name', 'Unknown')} {track_info.get('mode_name', '')}")
    print(f"Energy: {track_info.get('energy', 'Unknown'):.2f}")
    print(f"Danceability: {float(track_info.get('danceability', 0.0)):.2f}")
    print(f"Valence (positivity): {float(track_info.get('valence', 0.0)):.2f}")
    
    # Export results
    output_dir = "./analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    analyzer.visualize(output_dir=output_dir)
    
    json_path = analyzer.export_json(os.path.join(output_dir, "analysis_results.json"))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Visualization image: {os.path.join(output_dir, 'track_analysis.png')}")
    print(f"JSON data: {json_path}")
    
    return True

if __name__ == "__main__":
    test_basic_analysis()