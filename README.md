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

4. **How to Run**
   - Run the evaluation script:
      ```bash
     python main.py --dataset <name_of_the_dataset_you_want_to_evaluate>
     ```
   - This will evaluate all the models on the given dataset at an FMR of 0.01 and 0.001
   - The evaluation can also be specified for a specific FMR and model like this: 
      ```bash
     python main.py --dataset <name_of_the_dataset_you_want_to_evaluate> --fmr <fmr> --model <name_of_the_eval_model>
     ```
   - The datasets are named just like in the SYN-MAD22 directory (e.g. `MorDIFF_aligned`)
   - Embeddings and IDs will be saved in the `embeddings/` folder.
   - To generate our post-processed morphs run:
      ```bash
     python <name_of_the_attempt_script>.py
     ```
   - An example would be:
      ```bash
     python post_process_morphs.py
     ```
   - The post-processed datasets will be saved in the SYN-MAD22 directory
5. **Results**
   - The script prints MMPMR and IAPAR metrics for the selected model and dataset.

## License
This project is for research and educational purposes only. See LICENSE for details.
