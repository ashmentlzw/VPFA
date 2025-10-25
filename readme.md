1.dataset_preprocessing contains the preprocessing and partitioning of various datasets.
2.After obtaining each dataset, directly train the base model to get the base model weights. Then perform simple downsampling on the high-resolution images in the partitioned dataset, and use inference_features.py to infer features.
3.Use the obtained features to train via vpfa_train.py to get our trained VP.
4.Finally, apply the trained VP to the inference stage of the base model for feature post-processing, as in processor.py.
5.Additionally, the code for theoretical experimental verification of the existence of semantic directions in the feature space is in correlation_analysis.