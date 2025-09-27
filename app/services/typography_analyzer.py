import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pytesseract
from typing import Dict, List, Any, Tuple
import re
import os


class TypographyAnalyzer:
    """
    Advanced typography detection and analysis
    """

    def __init__(self):
        # Common font families for classification
        self.font_families = {
            "serif": ["Times", "Georgia", "Garamond", "Book Antiqua"],
            "sans-serif": ["Arial", "Helvetica", "Calibri", "Verdana", "Roboto"],
            "monospace": ["Courier", "Monaco", "Consolas", "Source Code Pro"],
            "script": ["Brush Script", "Lucida Handwriting", "Comic Sans"],
            "display": ["Impact", "Cooper Black", "Bebas Neue"]
        }

    def analyze_typography(self, image: Image.Image) -> Dict[str, Any]:
        """
        Comprehensive typography analysis
        """
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect text regions
        text_regions = self._detect_text_regions(cv_image)

        # Extract text content
        text_data = self._extract_text_content(cv_image, text_regions)

        # Analyze font characteristics
        font_analysis = self._analyze_font_characteristics(cv_image, text_regions)

        # Calculate readability scores
        readability = self._calculate_readability(text_data, cv_image)

        # Analyze typography hierarchy
        hierarchy = self._analyze_typography_hierarchy(text_regions, text_data)

        # Calculate overall typography score
        overall_score = self._calculate_typography_score(font_analysis, readability, hierarchy)

        return {
            "fonts_detected": font_analysis["detected_fonts"],
            "font_pairing_score": font_analysis["pairing_score"],
            "readability_score": readability["overall_score"],
            "text_hierarchy_score": hierarchy["hierarchy_score"],
            "overall_score": overall_score,
            "text_regions": text_regions,
            "detailed_analysis": {
                "font_analysis": font_analysis,
                "readability": readability,
                "hierarchy": hierarchy
            }
        }

    def _detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions using advanced computer vision techniques
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Method 1: MSER (Maximally Stable Extremal Regions)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)

            # Filter by size and aspect ratio
            if (w > 10 and h > 5 and
                w < image.shape[1] * 0.8 and h < image.shape[0] * 0.5 and
                0.1 < w/h < 20):  # Aspect ratio for text

                text_regions.append({
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "aspect_ratio": w / h,
                    "confidence": 0.8  # MSER confidence
                })

        # Method 2: Morphological operations for additional detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated = cv2.dilate(gray, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Similar filtering
            if (w > 15 and h > 8 and
                w < image.shape[1] * 0.7 and h < image.shape[0] * 0.4 and
                0.2 < w/h < 15):

                # Check if this region overlaps with existing ones
                overlap = False
                for existing in text_regions:
                    ex, ey, ew, eh = existing["bbox"]
                    if (abs(x - ex) < 20 and abs(y - ey) < 20 and
                        abs(w - ew) < 20 and abs(h - eh) < 20):
                        overlap = True
                        break

                if not overlap:
                    text_regions.append({
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "aspect_ratio": w / h,
                        "confidence": 0.6  # Morphological confidence
                    })

        # Sort by area (larger text regions first)
        text_regions.sort(key=lambda x: x["area"], reverse=True)

        return text_regions[:10]  # Limit to top 10 regions

    def _extract_text_content(self, image: np.ndarray, text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract text content from detected regions using OCR
        """
        text_data = []

        for region in text_regions:
            x, y, w, h = region["bbox"]

            # Extract region
            roi = image[y:y+h, x:x+w]

            try:
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(roi, config='--psm 8').strip()

                if text and len(text) > 1:  # Filter out single characters/noise
                    # Get additional details
                    data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)

                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0

                    text_data.append({
                        "text": text,
                        "bbox": region["bbox"],
                        "confidence": avg_confidence,
                        "word_count": len(text.split()),
                        "char_count": len(text),
                        "estimated_font_size": self._estimate_font_size(h)
                    })

            except Exception as e:
                # OCR failed for this region
                continue

        return text_data

    def _analyze_font_characteristics(self, image: np.ndarray, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze font characteristics and classify font types
        """
        detected_fonts = []
        font_sizes = []

        for region in text_regions:
            x, y, w, h = region["bbox"]

            # Estimate font size based on height
            font_size = self._estimate_font_size(h)
            font_sizes.append(font_size)

            # Extract region for analysis
            roi = image[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Classify font type based on visual characteristics
            font_type = self._classify_font_type(gray_roi)
            detected_fonts.append(font_type)

        # Calculate font pairing score
        unique_fonts = list(set(detected_fonts))
        pairing_score = self._calculate_font_pairing_score(unique_fonts, font_sizes)

        return {
            "detected_fonts": unique_fonts,
            "font_sizes": font_sizes,
            "pairing_score": pairing_score,
            "font_size_consistency": self._calculate_font_size_consistency(font_sizes)
        }

    def _classify_font_type(self, gray_roi: np.ndarray) -> str:
        """
        Classify font type based on visual characteristics
        """
        if gray_roi.size == 0:
            return "unknown"

        # Analyze stroke characteristics
        edges = cv2.Canny(gray_roi, 50, 150)

        # Count horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        h_count = np.sum(horizontal_lines > 0)
        v_count = np.sum(vertical_lines > 0)

        # Analyze stroke width variation
        stroke_variation = self._analyze_stroke_variation(gray_roi)

        # Simple classification based on characteristics
        if stroke_variation > 0.3:
            return "serif"
        elif h_count > v_count * 1.5:
            return "sans-serif"
        elif abs(h_count - v_count) < 10:
            return "monospace"
        else:
            return "sans-serif"  # Default to sans-serif

    def _analyze_stroke_variation(self, gray_roi: np.ndarray) -> float:
        """
        Analyze stroke width variation
        """
        if gray_roi.size == 0:
            return 0.0

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)

        # Calculate distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Calculate coefficient of variation for stroke widths
        stroke_widths = dist_transform[dist_transform > 0]
        if len(stroke_widths) == 0:
            return 0.0

        return np.std(stroke_widths) / np.mean(stroke_widths) if np.mean(stroke_widths) > 0 else 0.0

    def _estimate_font_size(self, height: int) -> int:
        """
        Estimate font size based on text height
        """
        # Rough estimation: font size ≈ height * 0.75
        return max(8, int(height * 0.75))

    def _calculate_font_pairing_score(self, fonts: List[str], sizes: List[int]) -> float:
        """
        Calculate font pairing quality score
        """
        if len(fonts) <= 1:
            return 100.0  # Single font always pairs well with itself

        # Penalize too many different fonts
        if len(fonts) > 3:
            return 30.0

        # Good pairing rules:
        # 1. Serif + Sans-serif = good
        # 2. Different weights of same family = good
        # 3. More than 2 font families = questionable

        score = 100.0

        # Check for good combinations
        if "serif" in fonts and "sans-serif" in fonts and len(fonts) == 2:
            score = 95.0  # Classic combination
        elif len(set(fonts)) == 1:
            score = 90.0  # Consistent font family
        elif len(fonts) == 2:
            score = 75.0  # Two fonts can work
        else:
            score = 50.0  # Multiple fonts are risky

        # Consider font size hierarchy
        if len(sizes) > 1:
            size_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else 1
            if 1.2 <= size_ratio <= 3.0:  # Good size hierarchy
                score += 10
            elif size_ratio > 3.0:  # Too much variation
                score -= 15

        return max(0, min(100, score))

    def _calculate_font_size_consistency(self, sizes: List[int]) -> float:
        """
        Calculate font size consistency score
        """
        if len(sizes) <= 1:
            return 100.0

        # Calculate coefficient of variation
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        if mean_size == 0:
            return 100.0

        cv = std_size / mean_size

        # Lower variation = higher consistency
        consistency_score = max(0, 100 - cv * 100)
        return consistency_score

    def _calculate_readability(self, text_data: List[Dict[str, Any]], image: np.ndarray) -> Dict[str, Any]:
        """
        Calculate text readability scores
        """
        if not text_data:
            return {"overall_score": 0, "details": {}}

        contrast_scores = []
        size_scores = []

        for text_info in text_data:
            x, y, w, h = text_info["bbox"]

            # Extract region
            roi = image[y:y+h, x:x+w]

            # Calculate contrast
            contrast_score = self._calculate_contrast_score(roi)
            contrast_scores.append(contrast_score)

            # Score font size
            font_size = text_info["estimated_font_size"]
            size_score = self._score_font_size(font_size)
            size_scores.append(size_score)

        # Calculate overall readability
        avg_contrast = np.mean(contrast_scores) if contrast_scores else 0
        avg_size_score = np.mean(size_scores) if size_scores else 0

        overall_score = (avg_contrast * 0.6 + avg_size_score * 0.4)

        return {
            "overall_score": overall_score,
            "contrast_score": avg_contrast,
            "size_score": avg_size_score,
            "details": {
                "individual_contrasts": contrast_scores,
                "individual_sizes": size_scores
            }
        }

    def _calculate_contrast_score(self, roi: np.ndarray) -> float:
        """
        Calculate WCAG-based contrast score
        """
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # Get text and background colors
        # Assume text is darker pixels, background is lighter
        text_pixels = gray[gray < np.percentile(gray, 50)]
        bg_pixels = gray[gray >= np.percentile(gray, 50)]

        if len(text_pixels) == 0 or len(bg_pixels) == 0:
            return 0.0

        text_luminance = np.mean(text_pixels) / 255.0
        bg_luminance = np.mean(bg_pixels) / 255.0

        # Calculate contrast ratio
        lighter = max(text_luminance, bg_luminance)
        darker = min(text_luminance, bg_luminance)

        contrast_ratio = (lighter + 0.05) / (darker + 0.05)

        # WCAG AA requires 4.5:1 for normal text, 3:1 for large text
        if contrast_ratio >= 4.5:
            return 100.0
        elif contrast_ratio >= 3.0:
            return 75.0
        elif contrast_ratio >= 2.0:
            return 50.0
        else:
            return 25.0

    def _score_font_size(self, font_size: int) -> float:
        """
        Score font size for readability
        """
        if font_size >= 16:  # Large text
            return 100.0
        elif font_size >= 14:  # Medium text
            return 85.0
        elif font_size >= 12:  # Small but readable
            return 70.0
        elif font_size >= 10:  # Very small
            return 45.0
        else:  # Too small
            return 20.0

    def _analyze_typography_hierarchy(self, text_regions: List[Dict[str, Any]], text_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze typography hierarchy effectiveness
        """
        if not text_data:
            return {"hierarchy_score": 0}

        # Sort by estimated font size (largest first)
        sorted_text = sorted(text_data, key=lambda x: x["estimated_font_size"], reverse=True)

        # Analyze size relationships
        sizes = [t["estimated_font_size"] for t in sorted_text]
        hierarchy_score = self._calculate_hierarchy_score(sizes)

        # Analyze spatial hierarchy (top to bottom, left to right)
        spatial_score = self._calculate_spatial_hierarchy_score(text_data)

        overall_hierarchy = (hierarchy_score * 0.7 + spatial_score * 0.3)

        return {
            "hierarchy_score": overall_hierarchy,
            "size_hierarchy": hierarchy_score,
            "spatial_hierarchy": spatial_score,
            "text_levels": len(set(sizes))
        }

    def _calculate_hierarchy_score(self, sizes: List[int]) -> float:
        """
        Calculate typography hierarchy score based on size relationships
        """
        if len(sizes) <= 1:
            return 80.0  # Single size is neutral

        # Check for clear hierarchy
        unique_sizes = sorted(set(sizes), reverse=True)

        if len(unique_sizes) == 1:
            return 60.0  # All same size - poor hierarchy

        # Ideal hierarchy has 2-4 distinct levels
        if 2 <= len(unique_sizes) <= 4:
            score = 90.0
        elif len(unique_sizes) > 4:
            score = 70.0  # Too many levels
        else:
            score = 80.0

        # Check size ratios
        ratios = []
        for i in range(len(unique_sizes) - 1):
            ratio = unique_sizes[i] / unique_sizes[i + 1]
            ratios.append(ratio)

        # Good ratios are between 1.2 and 2.5
        good_ratios = [r for r in ratios if 1.2 <= r <= 2.5]
        ratio_score = len(good_ratios) / len(ratios) * 100 if ratios else 100

        return (score + ratio_score) / 2

    def _calculate_spatial_hierarchy_score(self, text_data: List[Dict[str, Any]]) -> float:
        """
        Calculate spatial hierarchy score
        """
        if len(text_data) <= 1:
            return 80.0

        # Sort by Y position (top to bottom)
        sorted_by_position = sorted(text_data, key=lambda x: x["bbox"][1])

        # Check if larger fonts are generally above smaller ones
        position_size_correlation = 0
        for i, text in enumerate(sorted_by_position):
            expected_rank = i
            size_rank = sorted(text_data, key=lambda x: x["estimated_font_size"], reverse=True).index(text)

            # Calculate correlation between position and size
            position_size_correlation += abs(expected_rank - size_rank)

        # Normalize correlation score
        max_correlation = len(text_data) * (len(text_data) - 1) / 2
        correlation_score = max(0, 100 - (position_size_correlation / max_correlation * 100)) if max_correlation > 0 else 100

        return correlation_score

    def _calculate_typography_score(self, font_analysis: Dict[str, Any], readability: Dict[str, Any], hierarchy: Dict[str, Any]) -> float:
        """
        Calculate overall typography score
        """
        font_score = font_analysis["pairing_score"]
        readability_score = readability["overall_score"]
        hierarchy_score = hierarchy["hierarchy_score"]

        # Weighted average
        overall_score = (
            font_score * 0.3 +
            readability_score * 0.4 +
            hierarchy_score * 0.3
        )

        return overall_score