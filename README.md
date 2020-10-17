# SAR2SAR: a self-supervised despeckling algorithm for SAR images
Based on the work of Emanuele Dalsasso, Loïc Denis, Florence Tupin. [Link to Repo](https://github.com/emanueledalsasso/SAR2SAR)

The code is made available under the **GNU General Public License v3.0**: Copyright 2020, Emanuele Dalsasso, Loïc Denis, Florence Tupin, of LTCI research lab - Télécom ParisTech, an Institut Mines Télécom school.
All rights reserved.

Please note that the training set is only composed of **Sentinel-1** SAR images, thus this testing code is specific to this data.

## How to use the tool

1. Preprocess your image into '.npy' file. Check '00_Preprocessing.ipynb'.
2. Place your processed numpy data under the 'data' directory in the source folder
3. Run it through the model. Check '01_Interface.ipynb'.
4. Check for your denoised image under 'output' folder, on sucessful execution

Note: Use the 'test-data' branch to get test-data. The master branch doesn't include any testing data.

## Resources
- [Paper (ArXiv)](https://arxiv.org/abs/2006.15037)
The material is made available under the **GNU General Public License v3.0**: Copyright 2020, Emanuele Dalsasso, Loïc Denis, Florence Tupin, of LTCI research lab - Télécom ParisTech, an Institut Mines Télécom school.
All rights reserved.

To cite the article:

    @article{dalsasso2020sar2sar,
        title={{SAR2SAR}: a self-supervised despeckling algorithm for {SAR} images},
        author={Emanuele Dalsasso and Loïc Denis and Florence Tupin},
        journal={arXiv preprint arXiv:2006.15037},
        year={2020}
    }
