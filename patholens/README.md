# PathoLens

**Computational Histopathology Analysis for Cancer Detection**

PathoLens is a research demonstration tool that analyses H&E (Haematoxylin and Eosin) stained histopathology images to identify potential cancerous regions, estimate malignant cell density, and provide explainable insights for researchers and histopathologists.

> **Important:** This tool is designed for research purposes only and is not intended for clinical diagnosis or treatment decisions.

---

## Overview

PathoLens combines computer vision, deep learning, and interpretable AI to provide comprehensive histopathology analysis:

- **Cancer Probability Heatmaps** - Visual overlay showing regions with elevated malignancy probability
- **Nuclei Segmentation** - Detection and analysis of individual cell nuclei
- **Malignant Cell Estimation** - Quantification of suspected malignant cells
- **Tumour Grade Estimation** - Probabilistic grading based on histological features
- **AI-Powered Explanations** - Natural language interpretation of findings

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input H&E Image                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Patch Extraction                           │
│           (224×224 overlapping patches with tissue detection)   │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│    Cancer Classification  │   │      Nuclei Segmentation      │
│     (ResNet18-based)      │   │   (Colour Deconvolution +     │
│                           │   │    Morphological Operations)  │
└───────────────────────────┘   └───────────────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│   Heatmap Reconstruction  │   │   Nuclear Feature Extraction  │
│                           │   │   (Size, Shape, Intensity)    │
└───────────────────────────┘   └───────────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tumour Grade Estimation                      │
│        (Rule-based scoring using aggregated features)           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Explanation Generation                        │
│              (OpenAI API or template-based)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Results Display                           │
│    (Heatmap overlay, metrics, grade distribution, explanation)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone or download the repository:

```bash
cd patholens
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

---

## Configuration

### OpenAI API Key (Optional)

To enable AI-powered explanations, set your OpenAI API key:

**Option 1: Environment Variable**
```bash
export OPENAI_API_KEY=your-api-key-here
```

**Option 2: Create a `.env` file**
```
OPENAI_API_KEY=your-api-key-here
```

> Note: Without an API key, the tool will use template-based explanations which are still informative but less dynamic.

### Analysis Settings

Adjust these parameters in the sidebar:

| Setting | Description | Default |
|---------|-------------|---------|
| Patch Size | Size of analysis patches (pixels) | 224 |
| Stride | Step between patches (smaller = more overlap) | 112 |
| Minimum Tissue Fraction | Required tissue content per patch | 0.5 |
| Heatmap Opacity | Transparency of overlay | 0.5 |

---

## Adding Sample Slides

Place histopathology images in the `assets/` directory:

```bash
patholens/
└── assets/
    ├── sample_histo.png      # Default sample
    ├── my_slide_1.png
    └── my_slide_2.jpg
```

Supported formats: PNG, JPEG

### Recommended Image Sources

For testing, you can obtain sample H&E images from:

- [OpenSlide Test Data](https://openslide.org/demo/)
- [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/)
- [BACH Challenge Dataset](https://iciar2018-challenge.grand-challenge.org/)
- [Camelyon Dataset](https://camelyon17.grand-challenge.org/)

---

## Project Structure

```
patholens/
├── app.py                 # Streamlit frontend application
├── models/
│   ├── __init__.py
│   ├── load_model.py      # Model loading utilities
│   ├── classifier.py      # Cancer classification with Grad-CAM
│   ├── segmentation.py    # Nuclei segmentation pipeline
│   └── heatmap.py         # Heatmap generation and overlay
├── utils/
│   ├── __init__.py
│   ├── patching.py        # Patch extraction with tissue detection
│   ├── grading.py         # Tumour grade estimation
│   └── explanation.py     # OpenAI explanation generation
├── assets/
│   └── sample_histo.png   # Sample histopathology image
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## How It Works

### 1. Patch Extraction

The input image is divided into overlapping patches (default 224×224 pixels). A tissue detection algorithm filters out background regions, ensuring only tissue-containing patches are analysed.

### 2. Cancer Classification

Each patch is processed through a ResNet18-based classifier pretrained on ImageNet. The model outputs a probability score indicating likelihood of malignancy. Grad-CAM visualisation is available for deeper interpretability.

> **Note:** The current implementation uses ImageNet weights for demonstration. For clinical research, the model should be fine-tuned on histopathology datasets such as BACH or Camelyon.

### 3. Heatmap Generation

Patch-level predictions are aggregated to create a whole-slide probability map. Overlapping regions are averaged, and Gaussian smoothing is applied for visual continuity. The heatmap is overlaid on the original image using a diverging colour scale.

### 4. Nuclei Segmentation

H&E colour deconvolution separates the haematoxylin channel (staining nuclei blue/purple). Thresholding and morphological operations isolate individual nuclei. Watershed segmentation separates touching cells.

### 5. Grade Estimation

Aggregated features are used to estimate histological grade:

- **Grade 1** (Well-differentiated): Low nuclear pleomorphism, organised architecture
- **Grade 2** (Moderately-differentiated): Intermediate features
- **Grade 3** (Poorly-differentiated): High pleomorphism, disorganised architecture

The grading model uses:
- Mean cancer probability
- Nuclear size variation (pleomorphism)
- Nuclear shape irregularity (atypia)
- Texture heterogeneity

### 6. Explanation Generation

Findings are interpreted through either:
- **OpenAI API** (GPT-4o-mini): Dynamic, contextual explanations
- **Template System**: Structured explanations when API unavailable

---

## Limitations

This tool has several important limitations:

1. **Not for Clinical Use** - This is a research demonstration and has not been validated for clinical diagnosis

2. **Simplified Models** - The classifier uses ImageNet pretrained weights rather than histopathology-specific training

3. **Grading Approximation** - Tumour grade estimation is rule-based and does not incorporate all histological criteria

4. **Limited Validation** - The tool has not been validated against pathologist assessments

5. **Image Quality Sensitivity** - Results depend on image quality, staining consistency, and tissue preparation

6. **No Whole-Slide Support** - Currently designed for image patches rather than full WSI files

---

## Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| Streamlit | Web interface |
| PyTorch | Deep learning framework |
| torchvision | Pretrained models and transforms |
| scikit-image | Image processing and segmentation |
| OpenCV | Computer vision operations |
| Pillow | Image handling |
| matplotlib | Visualisation and colour maps |
| OpenAI | Explanation generation |

### Model Architecture

**Classifier:**
- Base: ResNet18 (pretrained on ImageNet)
- Modified head: Dropout → Linear(512, 256) → ReLU → Dropout → Linear(256, 2)
- Output: Binary classification (non-cancerous / cancerous)

**Segmentation:**
- Colour deconvolution for H&E separation
- Otsu thresholding on haematoxylin channel
- Morphological opening/closing
- Watershed for instance separation

---

## Future Enhancements

Potential improvements for production use:

- [ ] Fine-tune classifier on BACH/Camelyon datasets
- [ ] Integrate HoVerNet or StarDist for nuclei segmentation
- [ ] Add whole-slide image (WSI) support with OpenSlide
- [ ] Implement mitotic figure detection
- [ ] Add multi-class tumour type classification
- [ ] Include uncertainty quantification
- [ ] Support DICOM and other medical formats

---

## Acknowledgements

This tool builds upon research and open-source contributions from:

- PyTorch team for deep learning framework
- scikit-image for image processing algorithms
- Streamlit for rapid application development
- The histopathology research community for methodological foundations

---

## Licence

This project is provided for research and educational purposes. Please ensure appropriate ethics approval and data governance when using with patient data.

---

## Contact

For questions, issues, or contributions, please open an issue in the repository.

---

*PathoLens - Advancing computational histopathology research*
