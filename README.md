# Camera-Identification-CNN
Use CNN to extract features of cameras (from PRNU)

### Intuition
  The most significant feature we can use to identify the source camera is the ***Fixed Pattern Noise*** (FPN in abbreviation). Here we're trying to extract features from this noise, to be more precise, from PRNU, which is the ratio of the optical power on a pixel versus the electric output of it, and it's one part of FPN.
  
### Steps
- Apply the denoising filter on the img and then obtain the corresponding noise distribution for each image.
- Generate the fingerprints according to the noises.
- Crop the fingerprints into patches to make the input of CNN model.
- Train the CNN model (Inception v3 and 2 additional Fully Connected Layers) to predict the source cam from the patches of fingerprints (input as groups, average the output). 

### Difficulties
- Hard to find suitable (effective and efficient) denoising filters, which will largely influence the result because: $Noise=Img-Denoise(Img)$.
- Cropping can make use of local features, but also can lost possible non-local features, so how to better extract features from high-resolution images still need to be figured out.
