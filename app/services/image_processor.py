import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any
import base64
from io import BytesIO


class ImageProcessor:
    """
    Advanced image preprocessing and computer vision analysis
    """

    def __init__(self):
        pass

    def preprocess_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Comprehensive image preprocessing for analysis
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return {
            "original": image,
            "cv_image": cv_image,
            "dimensions": image.size,
            "enhanced": self._enhance_image(cv_image),
            "normalized": self._normalize_image(cv_image)
        }

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better analysis
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Reduce noise
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return enhanced

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for consistent analysis
        """
        # Convert to float32 and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        return normalized

    def extract_color_palette(self, image: Image.Image, num_colors: int = 8) -> List[str]:
        """
        Extract dominant color palette using K-means clustering
        """
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Reshape the image to be a list of pixels
        pixels = img_array.reshape(-1, 3)

        # Apply K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)

        # Convert to hex
        hex_colors = []
        for color in colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            hex_colors.append(hex_color)

        return hex_colors

    def analyze_color_harmony(self, colors: List[str]) -> Dict[str, Any]:
        """
        Analyze color harmony and relationships
        """
        # Convert hex to HSV for analysis
        hsv_colors = []
        for hex_color in colors:
            rgb = self._hex_to_rgb(hex_color)
            hsv = self._rgb_to_hsv(rgb)
            hsv_colors.append(hsv)

        # Analyze color scheme type
        scheme_type = self._detect_color_scheme(hsv_colors)

        # Calculate harmony score
        harmony_score = self._calculate_harmony_score(hsv_colors)

        return {
            "scheme_type": scheme_type,
            "harmony_score": harmony_score,
            "dominant_hue": self._get_dominant_hue(hsv_colors),
            "saturation_range": self._get_saturation_range(hsv_colors),
            "brightness_range": self._get_brightness_range(hsv_colors)
        }

    def detect_text_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect text regions using EAST text detector
        """
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Simple text detection using contours for now
        # In production, use EAST detector or EasyOCR
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size to likely text regions
            if 10 < w < image.width/2 and 5 < h < image.height/4:
                text_regions.append({
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "aspect_ratio": w / h
                })

        return text_regions

    def analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze layout and composition using computer vision
        """
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Calculate rule of thirds intersection points
        height, width = cv_image.shape[:2]
        third_x1, third_x2 = width // 3, 2 * width // 3
        third_y1, third_y2 = height // 3, 2 * height // 3

        intersection_points = [
            (third_x1, third_y1), (third_x2, third_y1),
            (third_x1, third_y2), (third_x2, third_y2)
        ]

        # Analyze visual weight distribution
        weight_map = self._calculate_visual_weight(cv_image)

        # Calculate balance score
        balance_score = self._calculate_balance_score(weight_map)

        # Detect focal points using saliency
        saliency_map = self._calculate_saliency(cv_image)
        focal_points = self._detect_focal_points(saliency_map)

        return {
            "rule_of_thirds_score": self._calculate_rule_of_thirds_score(cv_image, intersection_points),
            "balance_score": balance_score,
            "focal_points": focal_points,
            "visual_weight_distribution": self._analyze_weight_distribution(weight_map),
            "saliency_map": self._encode_saliency_map(saliency_map)
        }

    def generate_saliency_map(self, image: Image.Image) -> str:
        """
        Generate visual saliency map
        """
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        saliency_map = self._calculate_saliency(cv_image)
        return self._encode_saliency_map(saliency_map)

    # Helper methods
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        r, g, b = [x / 255.0 for x in rgb]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360

        # Saturation
        s = 0 if max_val == 0 else diff / max_val

        # Value
        v = max_val

        return (h, s, v)

    def _detect_color_scheme(self, hsv_colors: List[Tuple[float, float, float]]) -> str:
        """Detect color scheme type"""
        if len(hsv_colors) < 2:
            return "monochromatic"

        hues = [color[0] for color in hsv_colors]

        # Check for complementary (opposite hues)
        for i, hue1 in enumerate(hues):
            for hue2 in hues[i+1:]:
                diff = abs(hue1 - hue2)
                if 160 <= diff <= 200 or 160 <= (360 - diff) <= 200:
                    return "complementary"

        # Check for triadic (120 degrees apart)
        if len(hues) >= 3:
            for i, hue1 in enumerate(hues):
                for j, hue2 in enumerate(hues[i+1:], i+1):
                    for hue3 in hues[j+1:]:
                        angles = sorted([hue1, hue2, hue3])
                        if (abs(angles[1] - angles[0] - 120) < 30 and
                            abs(angles[2] - angles[1] - 120) < 30):
                            return "triadic"

        # Check for analogous (adjacent hues)
        min_hue = min(hues)
        max_hue = max(hues)
        if max_hue - min_hue <= 60:
            return "analogous"

        return "custom"

    def _calculate_harmony_score(self, hsv_colors: List[Tuple[float, float, float]]) -> float:
        """Calculate color harmony score"""
        if len(hsv_colors) < 2:
            return 100.0

        # Base score on saturation and value consistency
        saturations = [color[1] for color in hsv_colors]
        values = [color[2] for color in hsv_colors]

        sat_variance = np.var(saturations)
        val_variance = np.var(values)

        # Lower variance = higher harmony
        harmony_score = max(0, 100 - (sat_variance * 200 + val_variance * 200))

        return harmony_score

    def _get_dominant_hue(self, hsv_colors: List[Tuple[float, float, float]]) -> float:
        """Get dominant hue"""
        hues = [color[0] for color in hsv_colors]
        return np.mean(hues)

    def _get_saturation_range(self, hsv_colors: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """Get saturation range"""
        saturations = [color[1] for color in hsv_colors]
        return (min(saturations), max(saturations))

    def _get_brightness_range(self, hsv_colors: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """Get brightness range"""
        values = [color[2] for color in hsv_colors]
        return (min(values), max(values))

    def _calculate_visual_weight(self, image: np.ndarray) -> np.ndarray:
        """Calculate visual weight map"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection for structure weight
        edges = cv2.Canny(gray, 50, 150)

        # Brightness weight
        brightness_weight = gray.astype(np.float32) / 255.0

        # Combine weights
        weight_map = (edges.astype(np.float32) / 255.0) * 0.6 + brightness_weight * 0.4

        return weight_map

    def _calculate_balance_score(self, weight_map: np.ndarray) -> float:
        """Calculate visual balance score"""
        height, width = weight_map.shape

        # Split into quadrants
        mid_h, mid_w = height // 2, width // 2

        top_left = np.sum(weight_map[:mid_h, :mid_w])
        top_right = np.sum(weight_map[:mid_h, mid_w:])
        bottom_left = np.sum(weight_map[mid_h:, :mid_w])
        bottom_right = np.sum(weight_map[mid_h:, mid_w:])

        # Calculate balance
        horizontal_balance = abs(top_left + bottom_left - top_right - bottom_right)
        vertical_balance = abs(top_left + top_right - bottom_left - bottom_right)

        total_weight = np.sum(weight_map)
        if total_weight == 0:
            return 100.0

        balance_score = max(0, 100 - (horizontal_balance + vertical_balance) / total_weight * 100)
        return balance_score

    def _calculate_saliency(self, image: np.ndarray) -> np.ndarray:
        """Calculate saliency map using spectral residual"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Calculate spectral residual
        fft = np.fft.fft2(blurred)
        log_amplitude = np.log(np.abs(fft) + 1e-10)
        spectral_residual = log_amplitude - cv2.GaussianBlur(log_amplitude, (3, 3), 0)

        # Inverse FFT
        saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * np.angle(fft))))**2

        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        saliency = (saliency * 255).astype(np.uint8)

        return saliency

    def _detect_focal_points(self, saliency_map: np.ndarray) -> List[Dict[str, float]]:
        """Detect focal points from saliency map"""
        # Find peaks in saliency map
        threshold = np.percentile(saliency_map, 90)
        peaks = np.where(saliency_map > threshold)

        focal_points = []
        for y, x in zip(peaks[0], peaks[1]):
            focal_points.append({
                "x": float(x) / saliency_map.shape[1],
                "y": float(y) / saliency_map.shape[0],
                "strength": float(saliency_map[y, x]) / 255.0
            })

        # Sort by strength and return top 5
        focal_points.sort(key=lambda p: p["strength"], reverse=True)
        return focal_points[:5]

    def _calculate_rule_of_thirds_score(self, image: np.ndarray, intersection_points: List[Tuple[int, int]]) -> float:
        """Calculate how well the image follows rule of thirds"""
        saliency = self._calculate_saliency(image)

        scores = []
        for x, y in intersection_points:
            # Sample area around intersection point
            sample_size = 20
            x_start = max(0, x - sample_size)
            x_end = min(image.shape[1], x + sample_size)
            y_start = max(0, y - sample_size)
            y_end = min(image.shape[0], y + sample_size)

            region_score = np.mean(saliency[y_start:y_end, x_start:x_end])
            scores.append(region_score)

        # Average score normalized to 0-100
        return float(np.mean(scores)) / 255.0 * 100

    def _analyze_weight_distribution(self, weight_map: np.ndarray) -> Dict[str, float]:
        """Analyze visual weight distribution"""
        height, width = weight_map.shape

        # Divide into regions
        regions = {
            "top": np.sum(weight_map[:height//3, :]),
            "middle": np.sum(weight_map[height//3:2*height//3, :]),
            "bottom": np.sum(weight_map[2*height//3:, :]),
            "left": np.sum(weight_map[:, :width//3]),
            "center": np.sum(weight_map[:, width//3:2*width//3]),
            "right": np.sum(weight_map[:, 2*width//3:])
        }

        total_weight = np.sum(weight_map)
        if total_weight > 0:
            regions = {k: v/total_weight for k, v in regions.items()}

        return regions

    def _encode_saliency_map(self, saliency_map: np.ndarray) -> str:
        """Encode saliency map as base64 string"""
        # Apply colormap for visualization
        colored_saliency = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(colored_saliency, cv2.COLOR_BGR2RGB))

        # Encode as base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"