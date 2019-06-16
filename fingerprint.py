import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma


LG_SIZE = (4032, 3024)
MOTOX_SIZE = (3120, 4160)
MOTOD_SIZE = (2432, 4320)
MOTON_SIZE = (4130, 3088)
SONY_SIZE = (4000, 6000)
HTC_SIZE = (2688, 1520)
SGN_SIZE = (2322, 4128)
SGS_SIZE = (2322, 4128)
IP4_SIZE = (2448, 3264)
IP6_SIZE = (2448, 3264)
PATCH_SIZE = 299  # The input size of inception v3 network


size_dict = {'iP6': IP6_SIZE, 'iP4s': IP4_SIZE, 'GalaxyS4': SGS_SIZE, 'GalaxyN3': SGN_SIZE, 'MotoNex6': MOTOX_SIZE,
             'MotoMax': MOTOD_SIZE, 'MotoX': MOTOX_SIZE, 'HTC-1-M7': HTC_SIZE, 'Nex7': SONY_SIZE, 'LG5x': LG_SIZE}


def get_patches(img, rnum, cnum, patch_size):
    r_inds = random.sample(list(range(rnum)), 4)
    c_inds = random.sample(list(range(cnum)), 4)
    patches = []
    for r_ind, c_ind in zip(r_inds, c_inds):
        patches.append(img[r_ind*patch_size:(r_ind+1)*patch_size, c_ind*patch_size:(c_ind+1)*patch_size, :])
    return patches


def get_fingerprint(imgs, cam):
    numerator = 0
    dominator = 0
    cam_size = size_dict[cam]
    row_patches = cam_size[0] // PATCH_SIZE
    col_patches = cam_size[1] // PATCH_SIZE
    for img_path in imgs:
        '''
            Given that there are problems with the shape when using cv2.imread, use plt.imread instead, 
            which return a ndarray as well. Also notice that cv2 uses BGR, but plt here uses RGB.
            Considering we treat each channel equally, so I just did not modify other parts of code accordingly.
        '''
        img = plt.imread(img_path)
        if img.shape[:2] != cam_size:
            if (img.shape[0] * img.shape[1]) == (cam_size[0] * cam_size[1]):
                img = np.transpose(img, (1, 0, 2))
            else:
                continue
        patches = get_patches(img, row_patches, col_patches, PATCH_SIZE)
        for patch in patches:
            # Non-local denoising from: https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html
            sigma_est = np.mean(estimate_sigma(patch, multichannel=True))
            patch_kw = dict(patch_size=5,
                            patch_distance=6,
                            multichannel=True)
            dst = denoise_nl_means(patch, h=0.6 * sigma_est, sigma=sigma_est,
                                   fast_mode=True, **patch_kw)
            w = patch - dst
            numerator += w * patch
            dominator += patch ** 2

    return numerator / dominator
