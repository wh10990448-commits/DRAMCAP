"""Optional public COCO detector supplying normalized, class-labelled ROIs."""
from pathlib import Path

import torch
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                         fasterrcnn_mobilenet_v3_large_320_fpn)
from torchvision.transforms.functional import to_tensor


class RoiDetector:
    def __init__(self, model_root, device='cuda:0', threshold=0.3, max_rois=9):
        torch.hub.set_dir(str(Path(model_root)/'torch_hub'))
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()
        self.model.requires_grad_(False)
        self.categories = weights.meta['categories']
        self.device = device; self.threshold = threshold; self.max_rois = max_rois

    @torch.no_grad()
    def __call__(self, images):
        # Detector coordinates are in original image pixels, not EVA's 224px input.
        predictions = self.model([to_tensor(image).to(self.device) for image in images])
        result = []
        for image, prediction in zip(images, predictions):
            width, height = image.size; boxes = []
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if float(score) < self.threshold or len(boxes) >= self.max_rois:
                    break
                x1, y1, x2, y2 = box.tolist(); category = int(label)
                boxes.append({'bbox': [max(0,x1/width), max(0,y1/height), min(1,x2/width), min(1,y2/height)],
                              'category_id': category, 'name': self.categories[category],
                              'confidence': float(score), 'is_core': len(boxes) < 4})
            if not boxes:
                boxes = [{'bbox': [0,0,1,1], 'category_id': 0, 'name': 'scene', 'confidence': 1., 'is_core': True}]
            result.append(boxes)
        return result
