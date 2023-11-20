import cv2

import numpy as np


def refine_mask(
    mask: np.ndarray,
    area_threshold: float,
    mode: str = 'islands'
) -> np.ndarray:
    """
    Refines a mask by removing small islands or filling small holes based on area
    threshold.

    Parameters:
        mask (np.ndarray): Input binary mask.
        area_threshold (float): Threshold for relative area to remove or fill features.
        mode (str): Operation mode ('islands' for removing islands, 'holes' for filling
                    holes).

    Returns:
        np.ndarray: Refined binary mask.
    """
    mask = np.uint8(mask * 255)
    operation = cv2.RETR_EXTERNAL if mode == 'islands' else cv2.RETR_CCOMP
    contours, _ = cv2.findContours(
        mask, operation, cv2.CHAIN_APPROX_SIMPLE
    )
    total_area = cv2.countNonZero(mask) if mode == 'islands' else mask.size

    for contour in contours:
        area = cv2.contourArea(contour)
        relative_area = area / total_area
        if relative_area < area_threshold:
            cv2.drawContours(
                mask, [contour], -1, (0 if mode == 'islands' else 255), -1
            )

    return np.where(mask > 0, 1, 0)