# Example photos

Seven photos used by `examples/cats_dogs_gradcam.py` and `examples/cats_dogs_explain.ipynb`, and
attached to the `cats-dogs-photos` demo model as clickable examples on the dashboard's Explain tab.

All of them come from the **test split** of the Oxford-IIIT Pet dataset (the model never saw them
in training):

> O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, *Cats and Dogs*, IEEE Conference on
> Computer Vision and Pattern Recognition (CVPR), 2012. https://www.robots.ox.ac.uk/~vgg/data/pets/

The dataset's images are released under the
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
licence; these copies are redistributed under the same licence. They were re-encoded as JPEG
(quality 85) with the longer side at 448 px.

| File | Source image | Notes |
| --- | --- | --- |
| `cat_abyssinian.jpg` | `Abyssinian_216.jpg` | scaled up from 300 px (the Abyssinian photos are small) |
| `cat_siamese.jpg` | `Siamese_23.jpg` | |
| `cat_british_shorthair.jpg` | `British_Shorthair_25.jpg` | |
| `dog_pug.jpg` | `pug_19.jpg` | |
| `dog_samoyed.jpg` | `samoyed_199.jpg` | |
| `dog_beagle.jpg` | `beagle_26.jpg` | |
| `cat_and_dog.jpg` | `Siamese_23.jpg` (left) + `pug_19.jpg` (right) | 448 × 448 composite of two 224 × 448 portrait crops, so the model's square centre crop keeps both heads |
