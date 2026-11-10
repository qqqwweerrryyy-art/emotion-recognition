import torch, cv2, mediapipe, librosa, sklearn

print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Видеокарта:", torch.cuda.get_device_name(0))
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mediapipe.__version__)
print("librosa:", librosa.__version__)
print("scikit-learn:", sklearn.__version__)