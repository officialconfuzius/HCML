# Face Morphing Embedding & Evaluation Toolkit

This project provides a toolkit for extracting face embeddings from morphing datasets and evaluating morphing attack detection using state-of-the-art face recognition models.

## Features
- **Model Selection:** Easily switch between ElasticFaceArc, ElasticFaceCos, and CurricularFace models.
- **Flexible Dataset Support:** Process various morphing datasets (e.g., BonaFide_aligned, FaceMorpher_aligned, etc.) in the `SYN-MAD22` directory.
- **Batch Embedding Extraction:** Efficiently extract and save embeddings for large image sets.
- **Morphing Attack Evaluation:** Compute metrics such as MMPMR and IAPAR for morphing attack detection.

## Project Structure (May be adjusted, models and images are not being shared!)
```
main.py                  # Main script for embedding extraction and evaluation
backbones/               # Model backbone definitions
embeddings/              # Output folder for embeddings and IDs
models/                  # Pretrained model weights (.pth files)
SYN-MAD22/               # Morphing datasets (aligned images)
triplets/                # Triplet files for evaluation
```

## Quick Start
1. **Install Requirements**
   - Python 3.8+
   - PyTorch
   - OpenCV
   - tqdm
   - numpy

   You can install dependencies with:
   ```bash
   pip install torch opencv-python tqdm numpy
   ```

2. **Download Pretrained Models**
   - Place the required `.pth` model files in the `models/` directory.

3. **Prepare Datasets**
   - Place your morphing datasets in the `SYN-MAD22/` directory.

4. **Configure and Run**
   - Edit `main.py` to select the model and dataset:
     ```python
     model = 'ElasticFaceCos'  # or 'ElasticFaceArc', 'CurricularFace'
     morphing_dataset = 'FaceMorpher_aligned'  # or other dataset
     ```
   - Run the script:
     ```bash
     python main.py
     ```
   - Embeddings and IDs will be saved in the `embeddings/` folder.

5. **Results**
   - The script prints MMPMR and IAPAR metrics for the selected model and dataset.

## Customization
- To add new datasets, place them in `SYN-MAD22/` and update the `morphing_dataset` variable.
- To use different models, add their `.pth` files to `models/` and update the `model` variable.

## License
This project is for research and educational purposes only. See LICENSE for details.
