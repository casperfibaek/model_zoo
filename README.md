# model_zoo
Deep Learning model architectures for Earth Observation (PyTorch)

Download data from here:
    https://drive.google.com/file/d/1VkpSbLOReVXpT_IsFGjvxncv-ZsVTM26/view?usp=sharing

Unzip into the images folder.

EfficientNet
ResNets (ResNet-50, 101, & 152)
ResNext, ResNext v2, ResNext_inception_v2
InceptionNetworks (w/ or wo/ Residuals)
U-nets
Bottoms-up
VisionTransforms

TODO:
    For all networks implement an encoder-decoder network.
    The unet and classificiation network could be in the same file..
    Implement different scales of the same networks
        ResNet: tiny, nano, large, ...
        ResNetUnet: tiny, nano, large...
        
        ResNext: tiny, nano, large, ...
        ResNextUnet: tiny, nano, large, ...

        InceptionResNext: tiny, nano, large, ...
        InceptionResNextUnet: tiny, nano, large, ...

        InceptionResNet: tiny, nano, large, ...
        InceptionResNetUnet: tiny, nano, large, ...

        EfficientNet??
        SqueezeNet??
