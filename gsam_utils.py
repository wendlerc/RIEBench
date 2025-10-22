import cv2
import numpy as np
import torch
import supervision as sv
from torchvision.ops import box_convert
from typing import Tuple
import grounding_dino.groundingdino.datasets.transforms as T
from grounding_dino.groundingdino.util.inference import load_model, predict
import matplotlib.pyplot as plt

def resize_mask(mask, size=(16, 16), increase_mask=True, blur_ksize=31, blur_sigma=None):
    """
    Resizes the mask to the given size.
    Optionally increases the masked area by applying Gaussian blur before resizing.

    Args:
        mask: np.ndarray
        size: tuple, output size (h, w)
        increase_mask: bool, whether to increase the size of the masked area before resizing
        blur_ksize: int, kernel size for GaussianBlur (should be odd)
        blur_sigma: float or None, standard deviation for GaussianBlur.
            If None, selects sigma automatically as blur_ksize/6 following OpenCV guidance.

    Returns:
        bool np.ndarray of shape `size`
    """

    # How to choose sigma?
    # If blur_sigma is not provided (None), we use sigma = blur_ksize/6 as a practical rule of thumb.
    # See OpenCV docs: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    proc_mask = mask.astype(np.float32)
    if increase_mask:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        sigma = blur_sigma
        if sigma is None:
            sigma = blur_ksize / 6.0
        proc_mask = cv2.GaussianBlur(proc_mask, (blur_ksize, blur_ksize), sigma)
        proc_mask = proc_mask > 0

    small = cv2.resize(proc_mask.astype(np.float32), size, interpolation=cv2.INTER_LANCZOS4) > 0
    #plt.imshow(small)
    #plt.show()
    if small.astype(np.float32).sum() == 0:
        tmp = mask.reshape(16, 32, 16, 32).astype(np.float32)
        tmp = tmp.sum(axis=1)
        tmp = tmp.sum(axis=2)
        if (tmp >= 32*32).astype(np.float32).sum() == 0:
            print("trying to fix the mask...")
            # set the maximum gridcell to 1
            amax = tmp.argmax()
            tmp[np.unravel_index(amax, tmp.shape)] = 1
            return tmp.astype(bool)
    return small



def sam_mask(img, prompt, sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD):
    def load_image(img) -> Tuple[np.array, torch.Tensor]:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = img.convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    image_source, image = load_image(img)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    #if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        #torch.backends.cuda.matmul.allow_tf32 = True
        #torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    return detections, labels, annotated_frame