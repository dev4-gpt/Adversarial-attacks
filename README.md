# Adversarial-attacks
Jailbreaking Deep Models: Adversarial Attacks and Transferability Analysis
This repository contains the official code and report for the "Jailbreaking Deep Models" project, completed for the ECE-GY 6143: Deep Learning course at NYU Tandon School of Engineering. The project investigates the vulnerability of deep image classifiers to various adversarial attacks and analyzes the transferability of these attacks across different model architectures.

Abstract
This project investigates the vulnerability of deep image classifiers to adversarial attacks, focusing on a pretrained ResNet-34 model evaluated on a subset of the ImageNet-1K dataset. We implement and evaluate three types of attacks: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and targeted patch attacks. Each attack is constrained under an L-infinity budget, with perturbations designed to be visually imperceptible yet highly effective at degrading classification performance. We further explore the transferability of these attacks to a DenseNet-121 model. Our results highlight both the fragility and the cross-model vulnerability of modern deep vision systems under constrained adversarial conditions.

Key Results & Highlights
The primary goal was to degrade the performance of a ResNet-34 model on a 100-class subset of ImageNet. The results demonstrate a catastrophic success rate for the implemented attacks.

Metric

Top-1 Accuracy

Top-5 Accuracy

ResNet-34 Baseline (Clean)

70.40%

94.20%

After FGSM Attack (ϵ=0.02)

5.00%

35.40%

After PGD Attack (ϵ=0.02)

0.00%

0.80%

After Patch Attack (32x32)

58.80%

87.80%

Transferability to DenseNet-121
Attacks generated on ResNet-34 were tested against a DenseNet-121 model to assess transferability:

Baseline DenseNet-121 Accuracy: 70.80%

FGSM Transferred Attack: 58.80% (Moderate transferability)

PGD Transferred Attack: 58.40% (Strongest transferability)

Patch Transferred Attack: 69.40% (Minimal transferability)

Methodology
The project was structured into five distinct tasks:

Baseline Evaluation: Established the initial Top-1 and Top-5 accuracy of a pretrained ResNet-34 model on the clean test dataset.

Fast Gradient Sign Method (FGSM): Implemented a single-step gradient-based attack to create "Adversarial Test Set 1." This foundational attack served as a benchmark for performance degradation.

Projected Gradient Descent (PGD): Implemented a more powerful, iterative attack method with random initialization to create "Adversarial Test Set 2." PGD takes multiple smaller steps, projecting the result back into the allowed perturbation budget (ϵ-ball) at each iteration, resulting in a much more effective attack.

Patch-Based Attack: Implemented a localized PGD attack constrained to a random 32x32 patch on the image. This simulates a more realistic threat model where an attacker can only manipulate a small portion of the input.

Transferability Analysis: Evaluated the effectiveness of all three generated adversarial datasets against a different, "black-box" model (DenseNet-121) to determine if adversarial examples crafted for one architecture can fool another.

Getting Started
Prerequisites
Python 3.8+

PyTorch

Torchvision

NumPy

Matplotlib

Tqdm

A CUDA-enabled GPU is highly recommended for reasonable performance.

Installation
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Install the required packages:

pip install -r requirements.txt

Data Setup
This project requires a specific data and label structure to run correctly.

Dataset: Place the TestDataSet folder (containing the 100 class subfolders of images) inside the root directory or update the path in the notebook accordingly.

Label Mapping File: A crucial component is a JSON file that maps the dataset's folder names (n-synset IDs) to their true ImageNet integer indices.

Create a file named labels_list.json in the root directory.

This file must be a dictionary mapping stringified integer indices to a list containing the n-synset ID and the human-readable name.

Example Format:

{
  "0": ["n01440764", "tench"],
  "1": ["n01443537", "goldfish"],
  "401": ["n02672831", "accordion"],
  ...
}

Usage
The entire pipeline, from baseline evaluation to transferability analysis, is contained within the main Jupyter Notebook:

jupyter notebook "DL_final (1).ipynb"

Running the cells in order will:

Load the dataset and models.

Perform the baseline evaluation (Task 1).

Generate, save, and evaluate adversarial datasets for FGSM, PGD, and Patch attacks (Tasks 2, 3, 4).

Conduct the transferability analysis on DenseNet-121 (Task 5).

Display visualizations of original vs. adversarial images.

Generated adversarial datasets will be saved in the Adversarial_Datasets_Generated directory.

Example Visualizations
Below are examples of the model's predictions on an original image versus its adversarially perturbed counterpart. Note the imperceptible nature of the changes.

(Original Image)

Prediction: Siamese Cat (Correct)

(Adversarial Image after PGD)

Prediction: Guacamole (Incorrect)

References
Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2014)

Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2018)

Contributors
Aryaman Singh Dev

Nevin Mathews Kuruvilla

Rohan Subramaniam
