# Fair-Diffusion

Repository to the work: Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness

`test_notebook.ipynb` can be used to gain first insights into changing fair attributes during image generation.

Our results can be repoduced with the provided code. `generate_images.py` enables to generate images for our occupation list and `evaluate_images.ipynb` evaluates them. `CLIP_iEAT.ipynb` computes the iEAT to insight biases in CLIP.

The models stored in `dlib_models/` are taken from FairFace, which is used to classify generated images for gender.
