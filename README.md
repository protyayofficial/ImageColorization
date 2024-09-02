<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="20%" alt="IMAGECOLORIZATION-logo">
</p>
<p align="center">
    <h1 align="center">Image Colorization</h1>
</p>
<p align="center">
    <em><code>Transform black-and-white images into vibrant, lifelike color using GAN-based models.</code></em>
</p>
<p align="center">
    <img src="https://img.shields.io/github/license/protyayofficial/ImageColorization?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
    <img src="https://img.shields.io/github/last-commit/protyayofficial/ImageColorization?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/protyayofficial/ImageColorization?style=default&color=0080ff" alt="repo-top-language">
    <img src="https://img.shields.io/github/languages/count/protyayofficial/ImageColorization?style=default&color=0080ff" alt="repo-language-count">
</p>

<br>

##### Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Tests](#tests)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project aims to colorize black-and-white images using a GAN-based approach inspired by the Pix2Pix model. The goal is to restore colors to grayscale images by training deep neural networks capable of predicting color information from luminance data.

The project can be extended for various applications like:
- Enhancing old photographs
- Augmenting grayscale imagery in film restoration
- Assisting in medical imaging, satellite imagery, and other fields where colorization can enhance insights.

---

## Features

- **GAN-based Colorization**: (_Currently_) Uses the Pix2Pix model for image-to-image translation from grayscale to color.
- **Evaluation Metrics**: Includes FID, SSIM, and PSNR for evaluating the quality of generated images.
- **Flexible Dataloader**: Supports dynamic loading and preprocessing of image data.
- **Training and Testing Pipeline**: Comes with scripts to train and test the models on custom datasets.
- **Visualization Tools**: Easily visualize and save colorized results.

---

## Repository Structure

```sh
└── ImageColorization/
    ├── LICENSE
    ├── README.md
    ├── models
    │   └── pix2pix.py
    ├── scripts
    │   └── train_pix2pix.sh
    ├── test.py
    ├── train.py
    └── utils
        ├── pix2pixDataLoader.py
        ├── pix2pixInitWeights.py
        ├── pix2pixLoss.py
        └── pix2pixMetricMeters.py
```

---

## Modules

<details open><summary>Inference</summary>

| File | Summary |
| --- | --- |
| [test.py](https://github.com/protyayofficial/ImageColorization/blob/main/test.py) | This script is responsible for evaluating the trained image colorization model on the test dataset. It performs predictions and calculates various metrics to assess model performance. |

</details>

<details open><summary>Training the models</summary>

| File | Summary |
| --- | --- |
| [train.py](https://github.com/protyayofficial/ImageColorization/blob/main/train.py) | This script is the main training engine for the image colorization project. It handles the training loop, backpropagation, and saving of model checkpoints. |

</details>

<details open><summary>Models</summary>

| File | Summary |
| --- | --- |
| [pix2pix.py](https://github.com/protyayofficial/ImageColorization/blob/main/models/pix2pix.py) | This file contains the Pix2Pix architecture for image colorization, implementing both the generator (UNet-based) and discriminator (PatchGAN-based) models. It also includes the forward pass and loss calculations specific to the Pix2Pix model. |

</details>

<details open><summary>Scripts</summary>

| File | Summary |
| --- | --- |
| [train_pix2pix.sh](https://github.com/protyayofficial/ImageColorization/blob/main/scripts/train_pix2pix.sh) | This file contains the script to train the Pix2Pix model. |

</details>

<details open><summary>Utility Modules</summary>

| File | Summary |
| --- | --- |
| [pix2pixLoss.py](https://github.com/protyayofficial/ImageColorization/blob/main/utils/pix2pixLoss.py) | This utility handles the different loss functions used during training. It includes the L1 loss for the generator and the GAN loss for both generator and discriminator models. |
| [pix2pixInitWeights.py](https://github.com/protyayofficial/ImageColorization/blob/main/utils/pix2pixInitWeights.py) | This utility is responsible for weight initialization. It ensures that the model weights are initialized correctly, following best practices like Xavier initialization. |
| [pix2pixDataLoader.py](https://github.com/protyayofficial/ImageColorization/blob/main/utils/pix2pixDataLoader.py) | This file includes the data loading utilities used to create PyTorch DataLoaders for the training and validation datasets. It includes data augmentation techniques and handles conversion to LAB color space. |
| [pix2pixMetricMeters.py](https://github.com/protyayofficial/ImageColorization/blob/main/utils/pix2pixMetricMeters.py) | This utility provides functionality for tracking various metrics during training and validation, such as accuracy, PSNR (Peak Signal-to-Noise Ratio), and other performance indicators. |

</details>

---

##  Getting Started

###  Prerequisites

- **Python**: `version 3.11`

###  Installation

Build the project from source:

1. Clone the ImageColorization repository:
```sh
❯ git clone https://github.com/protyayofficial/ImageColorization
```

2. Navigate to the project directory:
```sh
❯ cd ImageColorization
```

3. Install the required dependencies:
```sh
❯ pip install -r requirements.txt
```

###  Usage

To run the project, execute the following command:

```sh
❯ chmod +x scripts/train_{model_name}.sh 
❯ scripts/train_{model_name}.sh
```

For example: 

```sh
❯ chmod +x scripts/train_pix2pix.sh 
❯ scripts/train_pix2pix.sh
```

###  Tests

Execute the test suite using the following command:

```sh
❯ python test.py
```

---

## Project Roadmap

- [X] **`Task 1`**: Downloaded the COCO 2017 dataset and prepared it for training and evaluation.
- [X] **`Task 2`**: Built a robust data loading pipeline with support for LAB color space and essential data augmentation techniques.
- [X] **`Task 3`**: Developed custom inference files to facilitate metric logging and visualizations, ensuring effective model evaluation.
- [X] **`Task 4`**: Designed and implemented the training loop, with accurate loss calculation, backpropagation, and regular model checkpointing.
- [X] **`Task 5`**: Developed a baseline Pix2Pix model for image colorization, establishing the foundation for future model improvements.
- [X] **`Task 6`**: Integrated evaluation metrics like SSIM, PSNR, and FID, enabling rigorous testing of the model's performance against real images from the dataset.
- [X] **`Task 7`**: Implemented pretrained backbone unet using fastai since the training images are less than what was used in the original literature.
- [ ] **`Task 8`**: Enhance the architecture to achieve more realistic and vivid image colorization using advanced GAN methodologies and novel techniques.
- [ ] **`Task 9`**: Conduct extensive hyperparameter tuning to optimize model performance and achieve better convergence.
- [ ] **`Task 10`**: Deploy the project as a fully functional web application, allowing users to upload grayscale images and receive colorized outputs, utilizing React, Vite, and Tailwind CSS.

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/protyayofficial/ImageColorization/issues)**: Submit bugs found or log feature requests for the `ImageColorization` project.
- **[Submit Pull Requests](https://github.com/protyayofficial/ImageColorization/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/protyayofficial/ImageColorization/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/protyayofficial/ImageColorization
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/protyayofficial/ImageColorization/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=protyayofficial/ImageColorization">
   </a>
</p>
</details>

---

##  License

This project is protected under the [MIT License](https://choosealicense.com/licenses/mit/) License. For more details, refer to the [LICENSE](./LICENSE) file.

---

## Acknowledgments

- This project is inspired by the pioneering work of **Isola _et al._**, whose **Pix2Pix model** has revolutionized the field of image-to-image translation using conditional GANs. Their seminal paper, ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/abs/1611.07004), played a crucial role in shaping the techniques I used for image colorization in this project. I also appreciate the comprehensive [GitHub repository associated with their paper](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which was invaluable for understanding and implementing the GAN architecture. Some code was adapted and simplified from their repository to better fit the requirements of this project.

- A special thank you goes to **Moein Shariatnia**, whose insightful article, ["Colorizing Black & White Images with U-Net and Conditional GAN"](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8), offered practical guidance for combining U-Net architectures with GANs in the context of image colorization. His tutorial was instrumental in refining the techniques employed in this project. You can explore more of his work through his [GitHub](https://github.com/moein-shariatnia).
---
