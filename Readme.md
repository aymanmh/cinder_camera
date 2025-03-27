# Webcam emotion detector using ViT

#### Apply models from https://huggingface.co/Xenova/facial_emotions_image_detection to webcam using onnx, cinder framework and opencv, inference only no training.

### Before building:
- Reinstall Microsoft.ML.OnnxRuntime.DirectML from Nuget package manager
- Update the 'include' and 'library' paths for OpenCV (from project settings).
- Download the [model](https://huggingface.co/Xenova/facial_emotions_image_detection/tree/main/onnx) and copy it to assets folder (update the name in code if different)
