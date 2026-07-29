from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Put the downloaded/raw dataset in this directory.
DATASET_DIR = PROJECT_DIR / "intel_robotic_welding_dataset"

# Feature extraction outputs.
AUDIO_OUTPUT_DIR = PROJECT_DIR / "Audio_process_output"
IMAGE_OUTPUT_DIR = PROJECT_DIR / "Image_process_output"
VIDEO_OUTPUT_DIR = PROJECT_DIR / "Video_process_output"
SENSOR_OUTPUT_DIR = PROJECT_DIR / "Sensor_process_output"

# Multimodal fusion and dimensionality-reduction outputs.
FUSED_VECTORS_DIR = PROJECT_DIR / "Fused_multimodal_vectors"
PCA_OUTPUT_DIR = PROJECT_DIR / "Fused_vectors_pca_32d"
TSNE_OUTPUT_DIR = PROJECT_DIR / "Fused_vectors_tsne_32d"
UMAP_OUTPUT_DIR = PROJECT_DIR / "Fused_vectors_umap_32d"
