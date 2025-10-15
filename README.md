Repository for CSC 3730 Fall 2025
=
Project using StarDist on SOMETHING. We used computer vision to count SOMETHING in images.

Made by myself and OTHERS

[StarDist Python Library](https://stardist.net/)

[ImageJ/Labkit](https://imagej.net/plugins/labkit/) - Used for annotations.

[Kaggle Dataset used](https://www.kaggle.com/datasets/raj123verma/seeds-counting?resource=download)

IMPORTANT NOTES: 
All of our training was done on a x64 Windows 11 system.

The GPU you use must be CUDA compatible. here is a link to see the list: https://developer.nvidia.com/cuda-gpus

The environment we used was created in Anaconda: https://www.anaconda.com/download/success

If you want to see the final results without making your own model, head to the completed model with results section below, otherwise start here.
=
You need to create a new environment and add the required libraries. This will be accomplished by the included environment.yml file.

**First**, clone the repository.

**Second**, open Anaconda and navigate to the folder containing the repository.

**Third**, use the command:
>conda env create -f environment.yml

Once that is finished, you need to activate the new environment. Use the command:
>conda activate seeds

To check that the process worked correctly, use the command:
>conda list

The quickest way to check is to look for the Stardist, cudatoolkit, and cudnn libraries.

Please go to **STEP FIVE:** below and continue there.

COMPLETED MODEL WITH RESULTS
=
If you would like to only see the finished product, look at the folders **FINISHEDMODEL** and **FINISHEDPREDICTIONS**.

The **FINISHEDMODEL** folder contains the model itself as well as images showing what the program selected as the testing/training/validation set. The selected images will always be the same unless you change the seed number inside the code.

The **FINISHEDPREDICTIONS** folder contains the images we selected to use to test our model and the results of all three predictions. The results are quite good.

The images used for training will be in the images folder. Their associated masks are in the masks folder. The masks are in .tif format, so they are not easily viewable. We used ImageJ to view them.

This concludes the results section.

CREATING YOUR OWN MODEL
=
**DUE TO GITHUB'S 100MB FILE SIZE LIMIT, THE METHOD BELOW WILL NOT WORK FOR STEPS 1-4. FOLLOW THE STEPS ABOVE FIRST.**

**Step One:**
Unzip and place the seedproject folder anywhere. If you placed it on the desktop, the directory should look similar to this
C:\Users\angel\Desktop\seedproject

**Step Two:**
Open Anaconda prompt and navigate to the seedproject folder.
"cd C:\Users\angel\Desktop\seedproject"
as an example.

**Step Three:**
Use the command(the period at the end is not a typo):
"tar -xzf seedproject.tar.gz -C ."
This will extract the environment in which we used. Could take a minute or two.

**Step Four:**
We need to acivate the environment by using the command:
"conda activate path\to\seedproject"
For example,
"conda activate C:\Users\angel\Desktop\seedproject"
***
**STEP FIVE:**
Use the command:
>python training.py --epochs 10 --dataset_size 6

This will start the training process. We used 10 epochs to save time. Our actual model used 40. Please don't change the epochs as the next part won't work without modifications to the code. This will not produce a great model by any means, but this is to show how the process works. Assuming everything goes to plan, each epoch could take between 30 to 60 seconds.

**STEP SIX:**
Once training has completed, you will get a message saying "Training is complete." Next, use the command:
>python prediction_test.py

This will use the model you just created to predict on three separate images. The images and their predictions can be found in the prediction folder once the prediction_test script is done. You should get a "Prediction complete." message. Be warned, the model may not produce results with the epochs being so small.

This concludes the make your own model section.
