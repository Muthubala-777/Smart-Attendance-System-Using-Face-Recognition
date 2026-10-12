from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Standard 112x112 5-point landmark canonical template for ArcFace alignment
# Point 0: Left Eye (image coordinates, x=38.3)
# Point 1: Right Eye (image coordinates, x=73.5)
# Point 2: Nose Tip (image coordinates, x=56.0, y=71.7)
# Point 3: Left Mouth Corner (image coordinates, x=41.5, y=92.4)
# Point 4: Right Mouth Corner (image coordinates, x=70.7, y=92.2)
ARCFACE_CANONICAL_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def umeyama_similarity_2d(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """
    Computes optimal 2D similarity transform (rotation, uniform scale, translation)
    mapping src to dst using closed-form Umeyama SVD formulation.
    Eliminates random sampling instability and guarantees identical normalization
    between enrollment and recognition frames.
    """
    num, dim = src.shape
    if num < 3 or dim != 2:
        return None

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return None
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) < 0:
            T[:dim, :dim] = np.dot(U, np.dot(np.diag([1, -1]), V))
        else:
            T[:dim, :dim] = np.dot(U, V)
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    src_var = (src_demean ** 2).sum(axis=1).mean()
    if src_var < 1e-8:
        return None

    scale = 1.0 / src_var * np.dot(S, d)
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean)
    T[:dim, :dim] *= scale
    return T[:2, :].astype(np.float32)


def preprocess_face(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    landmarks: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Standardized, Reusable Face Preprocessing Function.
    Used identically across:
    1. Student Enrollment
    2. Reference Photo Generation
    3. Live Camera Attendance Recognition
    4. Validation & Evaluation Benchmarks

    Steps:
    1. Calculate eye/nose landmark geometry.
    2. Compute closed-form Umeyama similarity/affine transform.
    3. Warp face to canonical frontal orientation preserving aspect ratio.
    4. Normalize dimensions to output_size (default 112x112).
    5. Return aligned normalized BGR face array.
    """
    if image is None or image.size == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    h, w = image.shape[:2]
    if bbox is None:
        bbox = (0, 0, w, h)

    template = ARCFACE_CANONICAL_5PTS if output_size == (112, 112) else (
        ARCFACE_CANONICAL_5PTS * np.array([output_size[0] / 112.0, output_size[1] / 112.0], dtype=np.float32)
    )

    if landmarks is not None and landmarks.shape == (5, 2):
        try:
            src_pts = landmarks.copy().astype(np.float32)
            # Ensure left eye (smaller x in image space) is index 0 and right eye (larger x) is index 1
            if src_pts[0, 0] > src_pts[1, 0]:
                src_pts[[0, 1]] = src_pts[[1, 0]]
                src_pts[[3, 4]] = src_pts[[4, 3]]

            # 1. Primary: Closed-form Umeyama analytical similarity transform
            tfm = umeyama_similarity_2d(src_pts, template)
            if tfm is None:
                tfm, _ = cv2.estimateAffinePartial2D(src_pts, template, method=cv2.LMEDS)

            if tfm is not None:
                det_val = tfm[0, 0] * tfm[1, 1] - tfm[0, 1] * tfm[1, 0]
                scale = np.sqrt(abs(det_val))
                if 0.10 < scale < 10.0:
                    aligned = cv2.warpAffine(
                        image,
                        tfm,
                        output_size,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101
                    )
                    return aligned
        except Exception:
            pass

    return _crop_centered_square(image, bbox, output_size)


def _crop_centered_square(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Extracts a centered, square crop around the face bounding box with balanced padding,
    preventing any aspect ratio distortion when resizing to output_size.
    """
    h, w = image.shape[:2]
    fx, fy, fw, fh = bbox

    # Calculate face center
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0

    # Determine square side length with 25% margin for hair/chin context
    side = max(fw, fh) * 1.25

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(x1 + side))
    y2 = int(round(y1 + side))

    # Check if padding is needed
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded_img = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(128, 128, 128)
        )
        crop_x1 = x1 + pad_left
        crop_y1 = y1 + pad_top
        crop_x2 = x2 + pad_left
        crop_y2 = y2 + pad_top
        cropped = padded_img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        cropped = image[y1:y2, x1:x2]

    if cropped is None or cropped.size == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)


class FaceAligner:
    """
    Performs precision 5-point similarity transformation on detected faces.
    Normalizes rotation, scale, and translation to standard 112x112 dimensions
    to ensure invariant feature extraction across enrollment and live recognition.
    Strictly preserves aspect ratio and prevents distortion or off-center cropping.
    """

    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        self.output_size = output_size

    def align_face(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return preprocess_face(image, bbox, landmarks, output_size=self.output_size)

    def generate_alignment_preview(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generates debugging/verification package containing original face crop,
        annotated landmarks visualization, and final aligned normalized face.
        """
        aligned = self.align_face(image, bbox, landmarks)
        h, w = image.shape[:2]
        fx, fy, fw, fh = bbox

        # Draw landmarks on a cropped view
        x1 = max(0, fx - 10)
        y1 = max(0, fy - 10)
        x2 = min(w, fx + fw + 10)
        y2 = min(h, fy + fh + 10)
        crop_viz = image[y1:y2, x1:x2].copy()

        if landmarks is not None and crop_viz.size > 0:
            for i, (lx, ly) in enumerate(landmarks):
                vx = int(lx - x1)
                vy = int(ly - y1)
                cv2.circle(crop_viz, (vx, vy), 3, (0, 255, 0), -1)

        return {
            "aligned_face": aligned,
            "landmark_viz": crop_viz,
            "has_landmarks": landmarks is not None
        }
