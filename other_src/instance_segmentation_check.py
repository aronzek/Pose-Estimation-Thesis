import rasterio
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData
import os

# Model setup
config = 'mmdetection/rtmdet-ins_tiny_8xb32-300e_coco.py'
checkpoint = 'mmdetection/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
model = init_detector(config, checkpoint, device='cpu')

container_like_ids = [7, 8]  # bus- 6, train - 7, truck - 8, boat - 9

def process_tiff_tiled_filtered(tiff_path, tile_size=1024, conf_thresh=0.05):
    with rasterio.open(tiff_path) as src:
        img = np.dstack([src.read(i) for i in (1, 2, 3)])
        img = img.astype(np.uint8)

        height, width, _ = img.shape
        vis_img = img.copy()

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = img[y:y + tile_size, x:x + tile_size]
                if tile.shape[0] < 10 or tile.shape[1] < 10:
                    continue  # skip tiny edge tiles

                result = inference_detector(model, tile)

                if hasattr(result, 'pred_instances'):
                    pred_instances: InstanceData = result.pred_instances
                    bboxes = pred_instances.bboxes.cpu().numpy()
                    masks = pred_instances.masks.cpu().numpy() if pred_instances.masks is not None else []
                    scores = pred_instances.scores.cpu().numpy()
                    labels = pred_instances.labels.cpu().numpy()

                    for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                        if label not in container_like_ids:
                            continue
                        if score < conf_thresh:
                            continue

                        x1, y1, x2, y2 = map(int, bbox)
                        abs_x1, abs_y1 = x + x1, y + y1
                        abs_x2, abs_y2 = x + x2, y + y2
                        cv2.rectangle(vis_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)

                        if i < len(masks):
                            mask = masks[i].astype(np.uint8) * 255
                            mask_resized = cv2.resize(mask, (tile.shape[1], tile.shape[0]))
                            color_mask = np.zeros_like(vis_img[y:y + tile_size, x:x + tile_size])
                            color_mask[mask_resized > 0] = (0, 0, 255)
                            vis_img[y:y + tile_size, x:x + tile_size] = cv2.addWeighted(
                                vis_img[y:y + tile_size, x:x + tile_size], 1.0, color_mask, 0.5, 0)

        os.makedirs('instance_seg_output', exist_ok=True)
        output_path = os.path.join(
            'instance_seg_output',
            os.path.basename(tiff_path).replace('.tif', '_vis_tiled_filtered.jpg')
        )
        cv2.imwrite(output_path, vis_img)
        print(f"✅ Saved filtered tiled visual result to: {output_path}")

# Original unfiltered
def process_tiff_tiled(tiff_path, tile_size=1024):
    with rasterio.open(tiff_path) as src:
        img = np.dstack([src.read(i) for i in (1, 2, 3)])
        img = img.astype(np.uint8)

        height, width, _ = img.shape
        vis_img = img.copy()

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = img[y:y+tile_size, x:x+tile_size]
                if tile.shape[0] < 10 or tile.shape[1] < 10:
                    continue

                result = inference_detector(model, tile)

                if hasattr(result, 'pred_instances'):
                    pred_instances: InstanceData = result.pred_instances
                    bboxes = pred_instances.bboxes.cpu().numpy()
                    masks = pred_instances.masks.cpu().numpy() if pred_instances.masks is not None else []

                    for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = map(int, bbox)
                        abs_x1, abs_y1 = x + x1, y + y1
                        abs_x2, abs_y2 = x + x2, y + y2
                        cv2.rectangle(vis_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)

                        if i < len(masks):
                            mask = masks[i].astype(np.uint8) * 255
                            mask_resized = cv2.resize(mask, (tile.shape[1], tile.shape[0]))
                            color_mask = np.zeros_like(vis_img[y:y+tile_size, x:x+tile_size])
                            color_mask[mask_resized > 0] = (0, 0, 255)
                            vis_img[y:y+tile_size, x:x+tile_size] = cv2.addWeighted(
                                vis_img[y:y+tile_size, x:x+tile_size], 1.0, color_mask, 0.5, 0)

    os.makedirs('instance_seg_output', exist_ok=True)
    output_path = os.path.join(
        'instance_seg_output',
        os.path.basename(tiff_path).replace('.tif', '_vis_tiled.jpg'))
    cv2.imwrite(output_path, vis_img)
    print(f"✅ Saved tiled visual result to: {output_path}")


#Downsampled Tiff method
def process_tiff(tiff_path, output_path='instance_seg_result.jpg'):
    with rasterio.open(tiff_path) as src:
        img = np.dstack([src.read(i) for i in (1, 2, 3)])
        img = img.astype(np.uint8)

        # Resize to avoid memory crash
        max_dim = 2048
        scale = min(max_dim / img.shape[1], max_dim / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        result = inference_detector(model, img)
        vis_img = img.copy()

        if hasattr(result, 'pred_instances'):
            pred_instances: InstanceData = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            masks = pred_instances.masks.cpu().numpy() if pred_instances.masks is not None else []

            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if i < len(masks):
                    mask = masks[i].astype(np.uint8) * 255
                    color_mask = np.zeros_like(vis_img)
                    color_mask[mask > 0] = (0, 0, 255)
                    vis_img = cv2.addWeighted(vis_img, 1.0, color_mask, 0.5, 0)

        os.makedirs('instance_seg_output', exist_ok=True)
        output_path = os.path.join('instance_seg_output', os.path.basename(tiff_path).replace('.tif', '_vis_downscaled.jpg'))
        cv2.imwrite(output_path, vis_img)
        print(f"✅ Saved downscaled visual result to: {output_path}")

# process_tiff_tiled('mission_1/data/south_terminal/odm_orthophoto.tif')
process_tiff_tiled_filtered('mission_1/data/south_terminal/odm_orthophoto.tif')


