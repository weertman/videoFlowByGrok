import cv2
import numpy as np

def compute_flow(prev, next, params):
    # Stub: Implement based on algorithm
    if params['algorithm'] == "Farneback":
        return cv2.calcOpticalFlowFarneback(
            prev, next, None,
            pyr_scale=0.5, levels=params['pyr_levels'], winsize=params['win_size'],
            iterations=params['iterations'], poly_n=5, poly_sigma=1.2, flags=0
        )
    elif params['algorithm'] == "Dense Optical Flow":
        # Use Farneback as default or DIS
        return cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(prev, next, None)
    else:
        raise ValueError("Unknown algorithm")

def compute_flowtrace(stack, params):
    if params['invert']:
        stack = 255 - stack  # Assuming uint8 grayscale.
    if params['bg_subtract']:
        bg = np.median(stack, axis=0)
        stack = stack - bg[None, :, :]
        stack = np.clip(stack, 0, 255)
    trace = np.max(stack, axis=0).astype(np.uint8)
    return trace  # 2D grayscale array.