import sys

print("Python:", sys.version)

pkgs = ['cv2', 'numpy', 'scipy', 'sklearn', 'dlib', 'face_recognition', 'onnxruntime', 'torch']
for p in pkgs:
    try:
        mod = __import__(p)
        print(f"  {p}: INSTALLED ({getattr(mod, '__version__', 'ok')})")
    except ImportError:
        print(f"  {p}: NOT installed")

import cv2
print("\nOpenCV features:")
print("  Has CascadeClassifier:", hasattr(cv2, 'CascadeClassifier'))
print("  Has FaceDetectorYN:", hasattr(cv2, 'FaceDetectorYN'))
print("  Has FaceRecognizerSF:", hasattr(cv2, 'FaceRecognizerSF'))
print("  Has face module:", hasattr(cv2, 'face'))
if hasattr(cv2, 'face'):
    print("  cv2.face dir:", [x for x in dir(cv2.face) if not x.startswith('_')])
