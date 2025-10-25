# Looking Alike From Far to Near: Enhancing Cross-Resolution Re-Identification via Feature Vector Panning(VPFA)

1. Dataset Preprocessing
The dataset_preprocessing directory contains code for the preprocessing and partitioning of various datasets.
2. Base Model Training & Feature Inference
After obtaining each dataset, train the base model directly to get the base model weights.
Perform simple downsampling on the high-resolution images in the partitioned dataset.
Use inference_features.py to run feature inference and obtain the required features.
3. VP Model Training
Use the features obtained in Step 2 to train via vpfa_train.py, and get the trained VP (Vector Projection) model.
4. Feature Post-Processing & Base Model Inference
Finally, apply the trained VP to the inference stage of the base model for feature post-processing.
Refer to processor.py for specific implementation details.
5. Additional Verification Code
The code for theoretical experimental verification of the existence of semantic directions in the feature space is stored in correlation_analysis.
