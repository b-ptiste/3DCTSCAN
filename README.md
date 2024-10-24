# 👨‍⚕️ Pulmonary Vessel Segmentation Project

The goal of this project is to develop an **automatic segmentation of pulmonary vessels** using classical filtering methods such as **Sato**, **Frangi**, and **Meijering**.

The dataset includes 3D CT scans of the lungs, along with lung segmentations. A reference segmentation of pulmonary vessels is provided to evaluate the results.

## 🧠 Study Overview

This study focuses on comparing different segmentation methods based on the following steps:

1. **Application of filters**: Apply Sato, Frangi, and Meijering filters on the CT scans.
2. **Thresholding**: Perform thresholding to segment the vessels.
3. **Evaluation with ROC curves**: Compare the methods using ROC and AUC curves.
4. **Visualization**: Visualize the results using 3D Slicer.

## 📒 Notebooks for Reproducibility

To get started and reproduce the results, please refer to the following notebooks:

- [🚀 This notebook clones the repository, selects the best models, and evaluates them](https://colab.research.google.com/drive/1ME6lzZiTPspZ3jhIus-1iJmS2C_iilZc?usp=sharing)
- [🔍 This notebook performs cross-validation and is more experimental](https://colab.research.google.com/drive/1FU18MRXjVBNAoOpf0wmrT7GCViMERity?usp=sharing)
- [⚡ This notebook was designed for a fast demo](https://colab.research.google.com/drive/1-201jSFsmnDfnysrBJLf7zDyZ70VeeDJ?usp=sharing)

**Note:** The datasets used are not publicly available.
