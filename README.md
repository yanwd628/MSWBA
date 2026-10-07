# MSWBA (under review)
---
MSWBA: Multi-Scale Wavelet Fusion with Bidirectional
Cross-Attention for Deepfake Detection

+ Our codes is based on widely used codebase [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file). Plealse follow it's offical guideness:

[Installation](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#1-installation)

[Download Data](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#2-download-data)

[Preprocessing (optional)](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#3-preprocessing-optional) 

[Rearrangement (optional)](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#4-rearrangement)

+ Training 

>     python training/train.py --detector_path ./training/config/config/detector/mswba_effnet_b5_detector.yaml

+ Evaluation

>     python training/test.py \
>     --detector_path ./training/config/config/detector/mswba_effnet_b5_detector.yaml \
>     --test_dataset "Celeb-DF-v2" \
>     --weights_path [path to weights]

PS: Our weights will be available.