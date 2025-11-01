Repository for CSC 3730 Fall 2025
=
Project using StarDist on SOMETHING. We used computer vision to count SOMETHING in images.

Made by myself, [Kim Nyugen](https://github.com/tngu589), and [Riley Richard]

[StarDist Python Library](https://stardist.net/)

[ImageJ/Labkit](https://imagej.net/plugins/labkit/) - Used for annotations.

[Kaggle Dataset used](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask)

IMPORTANT NOTES: 

All of our training was done on a x64 Windows 11 system.

If you would like to replicate our work, the GPU you use must be CUDA compatible to signficantly speed up the training process. Here is a link to see the list: https://developer.nvidia.com/cuda-gpus

The environment we used was created in Anaconda: https://www.anaconda.com/download/success

Setup with Anaconda
=
**First**, clone the repository.
>`git clone https://github.com/skylarwilson/CSC-3730`

**Second**, open Anaconda and navigate to the folder containing the repository.

**Third**, use the command:
>`conda env create -f environment.yml`

Once that is finished, you need to activate the new environment. Use the command:
>`conda activate project`

Setup with Docker
=
**First**
Download the image
>'in progress'

Running the training script
=
To run training script, use the following:
>`python training.py --epochs <integer> --dataset_size <integer> --testing_size <integer>`

After training, you can use the model to then predict on a test image.
>`python prediction_test.py`

