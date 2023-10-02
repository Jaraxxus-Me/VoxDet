import torch
import numpy as np
import os
from tqdm import tqdm

path = "data/OWID/P1"

obj_ids = os.listdir(path)

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n, h, w = masks.shape

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        if (mask==0).all():
            bounding_boxes[index, 0] = w//2 - 2
            bounding_boxes[index, 1] = h//2 - 2
            bounding_boxes[index, 2] = w//2 + 2
            bounding_boxes[index, 3] = h//2 + 2
        else:
            y, x = torch.where(mask != 0)
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

for obj_id in tqdm(obj_ids[6274:]):
    file = os.path.join(path, obj_id, 'info.npz')
    if not os.path.isfile(file):
        continue
    p1_data = np.load(file)
    rgb = torch.from_numpy(p1_data['rgb'].astype(np.float32))
    mask = torch.from_numpy(p1_data['mask'].astype(np.float32))[:,0]
    boxes_p1 = masks_to_boxes(mask)
    print("Checked! {}".format(obj_id))