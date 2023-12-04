# Alzheimer Disease Classification Project 🧠🤖

## Overview

This project aims to develop a classification system for Alzheimer's disease using two different modalities: MRI images and textual data from the DementiaBank dataset. The goal is to leverage machine learning techniques to predict the presence or absence of Alzheimer's disease based on these two distinct types of information.

## Files

### 1. `MRI_images.py` 🖼️

#### Description:

This Python script focuses on processing and analyzing MRI images to extract relevant features for Alzheimer's disease classification. It utilizes state-of-the-art machine learning algorithms and deep learning techniques to build a robust classification model.

#### Usage:

```bash
python MRI_images.py --input_data <path_to_MRI_data> --output_model <path_to_save_model>
```

## Parameters

- `--input_data`: Path to the directory containing MRI images data.
- `--output_model`: Path to save the trained model.

### 2. `text_data.py` 📝

#### Description

This Python script is designed to handle textual data extracted from the DementiaBank dataset. It employs natural language processing (NLP) techniques to preprocess and vectorize the text data, preparing it for classification.

#### Usage

```bash
python text_data.py --input_data <path_to_text_data> --output_model <path_to_save_model>

## Parameters

- `--input_data`: Path to the directory containing text data from the DementiaBank dataset.
- `--output_model`: Path to save the trained model.

## Dependencies 🛠️

Ensure you have the following dependencies installed before running the scripts:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Keras

You can install these dependencies using the following command:

```bash
pip install numpy pandas scikit-learn keras
```

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

