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


# Jailbreaking Deep Models: Adversarial Attacks and Transferability Analysis

A comprehensive investigation into the vulnerability of deep image classifiers, conducted for the ECE-GY 6143: Deep Learning course at NYU Tandon School of Engineering. This repository contains the project's official code, analysis, and final report.

---

## Abstract

This project provides a rigorous analysis of the vulnerability of deep image classifiers, focusing on a pretrained **ResNet-34** model evaluated against a 100-class subset of the ImageNet-1K dataset. We implement and evaluate three distinct adversarial attack methodologies: the foundational **Fast Gradient Sign Method (FGSM)**, the powerful iterative **Projected Gradient Descent (PGD)**, and a constrained **localized patch attack**. Each attack operates under a strict **L-infinity budget** ($\epsilon=0.02$ for global attacks), ensuring perturbations remain visually imperceptible. Furthermore, we investigate the **transferability** of these attacks to a different model architecture, **DenseNet-121**, to assess the generalization of adversarial vulnerabilities. Our findings demonstrate the profound fragility of modern deep vision systems and highlight the varying effectiveness and transferability of different attack vectors, contributing to the broader conversation on AI safety and robustness.

---

## Key Results & Highlights

Our experiments demonstrate the catastrophic success of well-crafted adversarial attacks. The primary target, a ResNet-34 model, was systematically degraded from a strong baseline performance to complete classification failure.

### **Table 1: ResNet-34 Performance Under Adversarial Attack**

| Metric                        | Top-1 Accuracy | Top-5 Accuracy |
| :---------------------------- | :------------: | :------------: |
| **Baseline (Clean Dataset)** |   **70.40%** |   **94.20%** |
| After FGSM Attack ($\epsilon=0.02$) |     5.00%      |     35.40%     |
| **After PGD Attack ($\epsilon=0.02$)** |   **0.00%** |   **0.80%** |
| After Patch Attack (32x32, $\epsilon=0.5$)    |     58.80%     |     87.80%     |

### **Transferability Analysis on DenseNet-121**

Adversarial examples generated for ResNet-34 were tested against an unseen DenseNet-121 model to evaluate attack transferability.

-   **DenseNet-121 Baseline Accuracy:** 70.80% (Top-1)
-   **Performance on PGD-perturbed data:** 58.40% (Top-1)
-   **Conclusion:** Global attacks like PGD demonstrate significant transferability, fooling a different architecture effectively. In contrast, localized patch attacks showed minimal transferability, indicating they exploit features more specific to the source model.

---

## Methodology

The project was executed through a systematic, multi-task approach to build a comprehensive understanding of adversarial phenomena.

### **Task 1: Baseline Evaluation**
First, we established the baseline performance of the pretrained ResNet-34 model on the clean test dataset. This provided the crucial benchmark against which all attacks were measured.

### **Task 2: Fast Gradient Sign Method (FGSM)**
We implemented the foundational single-step FGSM attack. This method calculates the gradient of the loss with respect to the input image and adds a small perturbation in the direction of the gradient's sign. This created our first adversarial dataset and demonstrated the basic principle of adversarial vulnerability.

### **Task 3: Projected Gradient Descent (PGD)**
To create a more powerful attack, we implemented PGD, an iterative method that is considered a first-order oracle for adversarial robustness. By taking multiple small steps and projecting the result back into the allowed perturbation budget ($\epsilon$-ball) at each iteration, PGD finds more optimal adversarial examples, leading to a much more significant degradation in model performance.

### **Task 4: Localized Patch-Based Attack**
We simulated a more physically plausible threat model by implementing a localized PGD attack. This attack constrains all perturbations to a small, randomly selected 32x32 patch on the image. A higher perturbation budget was used for the patch to offset the limited area of manipulation.

### **Task 5: Transferability Analysis**
Finally, we assessed the generalization of our created adversarial examples. All three generated datasets (from FGSM, PGD, and Patch attacks) were used to evaluate a pretrained DenseNet-121 model. This "black-box" test is critical for understanding if attacks exploit universal weaknesses in neural networks or just overfitting patterns of a specific model.

---

## Getting Started

Follow these instructions to set up the environment and replicate the project's results.

### **Prerequisites**
-   Python 3.8+
-   PyTorch & Torchvision
-   NumPy
-   Matplotlib
-   Tqdm
-   A CUDA-enabled GPU is strongly recommended.

### **Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **Data Setup**

Correct data and label setup is critical for the code to run.

1.  **Dataset:** Place the `TestDataSet` folder, which contains the 100 subfolders of images, in the project's root directory.

2.  **Label Mapping File:** This project requires a JSON file to map the dataset's folder names (which are ImageNet n-synset IDs) to their true integer class indices.
    -   Create a file named `labels_list.json` in the root directory.
    -   **This file must be a dictionary**, mapping stringified integer indices (e.g., "0", "1") to a list containing `["n_synset_id", "human_readable_name"]`.
    -   **Required Format Example:**
        ```json
        {
          "0": ["n01440764", "tench"],
          "1": ["n01443537", "goldfish"],
          "401": ["n02672831", "accordion"],
          ...
        }
        ```

---

## Usage

The entire pipeline is documented and executable within the primary Jupyter Notebook.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook "DL_final (1).ipynb"
    ```

2.  **Run the cells sequentially.** The notebook is structured to:
    -   Perform initial setup and data loading.
    -   Execute Task 1: Baseline Evaluation.
    -   Execute Task 2-4: Generate, save, and evaluate all three adversarial datasets.
    -   Execute Task 5: Conduct the transferability analysis.
    -   Display example visualizations of successful attacks.

All generated adversarial datasets will be saved to the `Adversarial_Datasets_Generated` directory for further analysis.

---

## Example Visualization: Original vs. PGD Attack

The following shows a sample image correctly classified by ResNet-34, and the corresponding adversarial image which, despite being visually identical to the human eye, is confidently misclassified.

**Original Image (`L-infinity` perturbation added)**
![An image of a siamese cat that has been subtly modified to fool an AI model.](https://i.imgur.com/uR1k3bN.png)
*Model Prediction: `Guacamole` (Incorrect)*

---

## References

-   Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples*. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
-   Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

---

## Contributors

-   **Aryaman Singh Dev**
-   **Nevin Mathews Kuruvilla**
-   **Rohan Subramaniam**

