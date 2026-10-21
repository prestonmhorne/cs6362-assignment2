# Phase 4: comparison with an existing implementation

## What exists
- I found no code release from the β-VAE authors (Higgins et al., DeepMind). The paper doesn't link any code, and
  DeepMind's official repo for this line of work, https://github.com/google-deepmind/dsprites-dataset, contains only
  the dataset (the .npz we use) and a loading notebook.
- Later implementations exist, e.g. Google Research's `disentanglement_lib` (Locatello et al., 2019) and community
  PyTorch repos (1Konny/Beta-VAE, YannDubs/disentangling-vae). None of them is by the paper's authors, and as far as
  I know they default to convolutional encoders/decoders and Adam, not this paper's FC 1200-1200 / 1200-1200-1200
  model with Adagrad. I did not read their code, so our implementation was not influenced by them.
- So there is no authors' code to compare against, and Phase 3 is our final version. Nothing was changed in Phase 4.

## Dataset check against the official repo
- The official notebook reads `imgs`, `latents_values` and `latents_classes` from
  `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`, with factor order (color, shape, scale, orientation, posX, posY)
  and sizes (1, 3, 6, 40, 32, 32) = 737,280 images. Our code uses the same file and keys.
