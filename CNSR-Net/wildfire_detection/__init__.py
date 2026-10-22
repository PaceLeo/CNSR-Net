"""CNSR-Net model variants for camera-aware wildfire detection."""

from .system import CNSRNet, DetectionResult
from .system_plus import CNSRNetPlus

__all__ = ["CNSRNet", "CNSRNetPlus", "DetectionResult"]
