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
    git clone [https://github.com/dev4-gpt/Adversarial-attacks.git]
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

The DL_project_2-2.pdf shows extensively evaluated images with pertubation added and incorrect model predictions.

---

## References

-   Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples*. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
-   Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

---

## Contributors

-   **Aryaman Singh Dev**
-   **Nevin Mathews Kuruvilla**
-   **Rohan Subramaniam**

