Repository for CSC 3730 Fall 2025
=
Project using StarDist on SOMETHING. We used computer vision to count SOMETHING in images.

Made by myself, [Kim Nyugen](https://github.com/tngu589), and [Riley Richard](https://github.com/rileythampersand)

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
>`conda activate cells`

Once activated, run the command:
>`cells train`

This can take a few minutes depending on if you have a GPU compatible with CUDA or not.

Once the training is done, use the command:
>`cells predict`

Once that is done, take a look inside predictions for an image that is produced by our model.
There you can see how well it did as well as a count of how many the model predicted.
