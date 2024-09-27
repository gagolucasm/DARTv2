# DARTv2: Deep Approximation of Retinal Traits

## Overview

**DARTv2** is a state-of-the-art tool for retinal vascular phenotyping, significantly improving upon the original DART model. It provides highly efficient, robust, and repeatable measurements of **Fractal Dimension (FD)** and **Vessel Density (VD)** from retinal colour fundus images. This tool is designed for easy integration and fast analysis, offering both a local GUI for batch processing and an accessible web interface.

### Keywords:
- Retinal image analysis
- Deep learning
- Robustness
- Fractal Dimension (FD)
- Vessel Density (VD)

## Paper

- **Title**: Self-consistent deep approximation of retinal traits for robust and highly efficient vascular phenotyping of retinal colour fundus images
- **Authors**: Lucas Gago, Beatriz Remeseiro, Laura Igual, Amos Storkey, Miguel O. Bernabeu, Justin Engelmann
- **Status**: Pending publication (available on [OpenReview](https://openreview.net/forum?id=HkVqbtphwf))

### Abstract
Retinal colour fundus images offer a fast, low-cost, non-invasive way of imaging the retinal vasculature, which provides critical insights into both ocular and systemic health. Traditional approaches to retinal vascular phenotyping rely on handcrafted, multi-step pipelines that are computationally intensive and sensitive to image quality issues. DARTv2 overcomes these limitations by leveraging a self-consistent deep learning model that is fast, robust, and repeatable. It enhances the original DART by adding **Vessel Density (VD)** as a new trait, incorporating additional augmentations, and improving repeatability through a self-consistency loss. DARTv2 demonstrates high agreement with the AutoMorph pipeline (Pearson 0.9392 for FD and 0.9612 for VD), is more robust than both AutoMorph and the original DART, and achieves a significant speed-up in processing.

## Features

- **Fast and Efficient**: 200x faster than AutoMorph and 4x faster than DART.
- **Highly Robust**: More resilient to image quality issues compared to traditional pipelines.
- **Repeatability**: Trained with a self-consistency loss for improved trait measurement consistency.
- **Batch Processing**: Supports drag-and-drop functionality for multiple images.
- **Results Export**: Outputs **Fractal Dimension (FD)** and **Vessel Density (VD)** as CSV, Excel, or TXT files.

## Installation and Usage

### Local Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/dartv2.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the GUI:
    ```bash
    streamlit run streamlit_dartv2_v3.py
    ```
4. Drag and drop multiple retinal images to begin analysis. The results will be saved in the desired format.

### Web Interface

DARTv2 is also accessible via a free web interface: [DARTv2 Web App](https://dartv2.streamlit.app/). Images are only stored in RAM during the session, ensuring data privacy. Simply upload your images, and download the results in your preferred format.

## Model Weights

The model weights are included in the repository for local inference.

## Citation

If you use DARTv2 in your research, please cite the pending publication:

@inproceedings{
gago2024selfconsistent,
title={Self-consistent deep approximation of retinal traits for robust and highly efficient vascular phenotyping of retinal colour fundus images},
author={Lucas Gago and Beatriz Remeseiro and Laura Igual and Amos Storkey and Miguel O. Bernabeu and Justin Engelmann},
booktitle={MICCAI Student Board EMERGE Workshop: Empowering MEdical image computing {\&} Research through early-career Expertise},
year={2024},
url={https://openreview.net/forum?id=HkVqbtphwf}
}

## Acknowledgments

We would like to thank the authors and contributors of the original DART model and the AutoMorph pipeline for their foundational work in this field.
