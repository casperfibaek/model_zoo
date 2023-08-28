# model_zoo
Deep Learning model architectures for Earth Observation (PyTorch)

Download data from here:
    https://drive.google.com/file/d/1VkpSbLOReVXpT_IsFGjvxncv-ZsVTM26/view?usp=sharing

Unzip into the images folder.

Training recipe:
    For patch based models:
        1. Train AE first with only flip and rotation augmentation.
        Use MSE or tiled MSE loss.
        Do this on a subset of the whole data (say 10%)

        2. Train the Masked AE with masking and flip and rotation augmentation.
        Use Tiled Mape as loss
        Do this on the whole dataset.
        If the dataset is so massive that epoching is not possible.
        Use DatasetRandomSampling and run the training in a while loop, saving a checkpoint every ~1000 iterations.
        This will allow you to resume training from the last checkpoint, and train for as long as you want. (fit permitting)

        3. Use the MAE model as encoder and train the downstream task with simple augmentations.
     