The .py files (e.g. TRAIN_16FEATURES_GITHUB.py) allow to train the networks and save reconstructed vibratory signals and latent representation.

They expect the vibratory signal in the folder "./LMT_FORMATTED/", in the working folder

Reconstructed signal and latent representations are saved as matlab arrays (i.e. ".mat" files) 

Output foders are created automatically. 

The "Code_LMT_XXFEATURES" contains the matlab arrays representing the latent representations  with size : time sample x features. XX could be 8, 16 or 32 features
The latent representations encoded by the VAE are saved in "./CodeVAE16D/"

The reconstructed signals are saved in the "./recontructed_LMT_XXFEATURES/" folders (XX could be 8, 16 or 32 features); in the "./ReconstructedVAE16D" for the VAE network. 
Their size is the one of the original signals

The trained full model is saved as "autoencoder_model_LMT_XXFEATURES.h" (XX could be 8, 16 or 32 features), as "vae_model.h" for the VAE
The trained encoder model (to produce the latent representation)  is saved as "encoder_model_LMT_XXFEATURES.h" (XX could be 8, 16 or 32 features), as "encoder_model.h" for the VAE 

The reconstructed signals and the latent representations are saved as matlab files indicated by progressive numbers. The "imageListLMT.txt" file lists the samples with their original names, in the order corresponding to those progressive numbers.

The matlab file "FORTMAT_LTM_DATABASE_GITHUB.m" takes as input the LTM database unzipped folder from: https://zeus.lmt.ei.tum.de/downloads/texture/download/LMT_108_SurfaceMaterials_Database_V1.1.zip and renders the processed files in the "./LMT_FORMATTED/" folder
The formatted data is high passed to removed frequencies below 10Hz, and normalized to force the full range between 0 and 1. This is done with the same linear transformation applied to all samples, which is reversed in Figure 1B
