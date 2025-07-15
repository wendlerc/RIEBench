import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2
from functools import partial
import json
from gsam_utils import sam_mask, resize_mask

logger = logging.getLogger(__name__)

code_to_block = {"18": "transformer.transformer_blocks.18"}

def add_activations(delta, module, input, output):
    if isinstance(output, torch.Tensor):
        return output + delta
    else:  
        return (output[0] + delta,)
    
@torch.no_grad()
def get_features_all_blocks(cache1, cache2, mask1, mask2, k_transfer=None, blocks_to_intervene=None, saes=None,
                            normalize=False):
    if saes is not None:
        raise ValueError("saes is not supported yet for flux")
    source_dict = {}
    target_dict = {}
    
    for idx, shortcut in enumerate(blocks_to_intervene): 
        # not sure whether this makes sense but afaik this is how the flux caching seems to work
        # instead of creating a cache dict, you have a cache list?
        block = code_to_block[shortcut]
        diff1 = cache1.image_activation[idx] 
        diff2 = cache2.image_activation[idx]
        source = diff1[:, mask1.flatten(), :]
        target = diff2[:, mask2.flatten(), :]

        source_dict[shortcut] = source.detach().cpu()
        target_dict[shortcut] = target.detach().cpu()

    return source_dict, target_dict


def setup_activation_interventions(blocks_to_intervene,
                                   source_dict,
                                   target_dict,
                                   mask1, 
                                   mask2, 
                                   m1, 
                                   stat, 
                                   subtract_target_add_source, 
                                   maintain_spatial_info, 
                                   mode,
                                   device,
                                   saes=None):
    if saes is not None:
        raise ValueError("saes is not supported yet for flux")
    interventions = {}
    for shortcut in blocks_to_intervene:
        # set up interventions 
        source = source_dict[shortcut]
        target = target_dict[shortcut]
        block = code_to_block[shortcut]
        logger.debug("Selecting best features...")
        if mode == "steering":
            # create the batch x feats x y x x tensor to add
            delta = torch.zeros((1, 4096, source.shape[2]), dtype=source.dtype, device=device)
            if subtract_target_add_source:
                if maintain_spatial_info:
                    delta[:, mask1.flatten(), :] += m1*source.mean(dim=0, keepdim=True).to(delta.device)
                    delta[:, mask2.flatten(), :] -= m1*target.mean(dim=0, keepdim=True).to(delta.device)
                else:
                    delta[:, mask1.flatten(), :] += m1*source.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).to(delta.device)
                    delta[:, mask2.flatten(), :] -= m1*target.mean(dim=0, keepdim=True).to(delta.device)
            else:
                delta[:, mask2.flatten(), :] += m1*source.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).to(delta.device)
                delta[:, mask2.flatten(), :] -= m1*target.mean(dim=0, keepdim=True).to(delta.device)
            f = partial(add_activations, delta)
            interventions[block] = f
        else:
            ValueError(f"Only supported mode currently is 'steering'.")
    return interventions

def run_feature_transport(prompt1, prompt2, gsam_prompt1, gsam_prompt2, pipe, grounding_model, sam2_predictor, saes=None,
         blocks_to_intervene=["18"], guidance=0.0, 
         n_steps=1, m1=1., k_transfer=10, stat="quantile", mode="sae",  
         use_source_mask_in_both=False, subtract_target_add_source=False,
         maintain_spatial_info=False, verbose=False,
         BOX_THRESHOLD=0.25, TEXT_THRESHOLD=0.25,
         result_name=None):
    assert blocks_to_intervene is not None, "blocks_to_intervene must be provided"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)    

    blocks = [code_to_block[shortcut] for shortcut in blocks_to_intervene]

    logger.debug("Generating images and caching activations...")
    seed = 42
    out1 =pipe.run(
        prompt1, 
        num_inference_steps=n_steps,
        width=1024,
        height=1024,
        cache_activations=True,
        guidance_scale=guidance,
        positions_to_cache=blocks,
        inverse=False,
        seed=seed
    )
    cache1 = pipe.activation_cache
    img1 = out1.images[0]

    # Generate second image and cache
    out2 = pipe.run(
        prompt2, 
        num_inference_steps=n_steps,
        width=1024,
        height=1024,
        cache_activations=True,
        guidance_scale=guidance,
        positions_to_cache=blocks,
        inverse=False,
        seed=seed
    )
    cache2 = pipe.activation_cache
    img2 = out2.images[0]

    dim = int(np.sqrt(cache1.image_residual[0].shape[1]))

    logger.debug("Running Grounded SAM on generated images...")
    if gsam_prompt1 == "#everything":
        mask1 = np.ones((dim, dim), dtype=bool)
    else:
        if "background" in gsam_prompt1:
            detections1, labels1, annotated_frame1 = sam_mask(img1, "foreground", sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask, size=(dim, dim)).astype(np.float32) for bigmask in detections1.mask]
            mask1 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
            mask1 = np.logical_not(mask1)
            if verbose:
                plt.imshow(mask1)
                plt.show()
        elif "~" in gsam_prompt1:
            detections1, labels1, annotated_frame1 = sam_mask(img1, gsam_prompt1.replace("~", ""), sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask, size=(dim, dim)).astype(np.float32) for bigmask in detections1.mask]
            mask1 = ~np.stack(masks, axis=0).sum(axis=0).astype(bool)
        else:
            detections1, labels1, annotated_frame1 = sam_mask(img1, gsam_prompt1, sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask, size=(dim, dim)).astype(np.float32) for bigmask in detections1.mask]
            mask1 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
    if gsam_prompt2 == "#everything":
        mask2 = np.ones((dim, dim), dtype=bool)
    else:
        if "background" in gsam_prompt2:
            detections2, labels2, annotated_frame2 = sam_mask(img2, "foreground", sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask, size=(dim, dim)).astype(np.float32) for bigmask in detections2.mask]
            mask2 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
            mask2 = np.logical_not(mask2)
            if verbose:
                plt.imshow(mask2)
                plt.show()
        elif "~" in gsam_prompt2:
            detections2, labels2, annotated_frame2 = sam_mask(img2, gsam_prompt2.replace("~", ""), sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask, size=(dim, dim)).astype(np.float32) for bigmask in detections2.mask]
            mask2 = ~np.stack(masks, axis=0).sum(axis=0).astype(bool)
        else:
            detections2, labels2, annotated_frame2 = sam_mask(img2, gsam_prompt2, sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask, size=(dim, dim)).astype(np.float32) for bigmask in detections2.mask]
            mask2 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
    if mask1.sum() == 0 or mask2.sum() == 0:
        raise ValueError("one of the masks is empty")
    if use_source_mask_in_both:
        detections2, labels2, annotated_frame2 = detections1, labels1, annotated_frame1
        mask2 = mask1
        gsam_prompt2 = gsam_prompt1
    if verbose:
        plt.imshow(mask1)
        plt.show()
        plt.imshow(mask2)
        plt.show()
    logger.debug("Extracting latents and encoding features...")
    
    assert mode == "steering", "for flux currently only steering is supported"

    source_dict, target_dict = \
        get_features_all_blocks(cache1, cache2, mask1, mask2, k_transfer=k_transfer, blocks_to_intervene=blocks_to_intervene, saes=saes)

    interventions = setup_activation_interventions(blocks_to_intervene, 
                                                    source_dict, 
                                                    target_dict, 
                                                    mask1, 
                                                    mask2, 
                                                    m1, 
                                                    stat, 
                                                    subtract_target_add_source, 
                                                    maintain_spatial_info, 
                                                    mode,
                                                    device,
                                                    saes=saes)
                
    hook = list(interventions.values())[0] # this is ugly as hell but the api for flux is different from our sdxl one...
    result = pipe.run_with_edit(
                    prompt2,
                    seed=seed,
                    num_inference_steps=int(n_steps),
                    edit_fn=lambda input, output: hook(None, input, output), 
                    layers_for_edit_fn=[i for i in range(int(blocks_to_intervene[0]), 57)],
                    stream="image").images[0]

    # make a result figure that shows the images with masks and the intervened image
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image 1 with mask from prompt 1
    if "~" not in gsam_prompt1 and "background" not in gsam_prompt1 and gsam_prompt1 != "#everything":
        axs[0].imshow(annotated_frame1)
    else:
        if "background" in gsam_prompt1 or "~" in gsam_prompt1:
            mask1 = cv2.resize(mask1.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # Plot the base image first
            axs[0].imshow(img1)
            # Then overlay the mask with higher alpha for visibility
            axs[0].imshow(mask1, alpha=0.5)
        else:
            axs[0].imshow(img1)
    axs[0].set_title(f"{prompt1}")
    axs[0].axis('off')
    
    # Image 2 with mask from prompt 2
    if "~" not in gsam_prompt2 and "background" not in gsam_prompt2 and gsam_prompt2 != "#everything":
        axs[1].imshow(annotated_frame2)
    else:
        if "background" in gsam_prompt2 or "~" in gsam_prompt2:
            mask2 = cv2.resize(mask2.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # Plot the base image first
            axs[1].imshow(img2)
            # Then overlay the mask with higher alpha for visibility
            axs[1].imshow(mask2, alpha=0.5)
        else:
            axs[1].imshow(img2)
    axs[1].set_title(f"{prompt2}")
    axs[1].axis('off')
    
    # Intervened result image
    axs[2].imshow(result)
    axs[2].axis('off')

    if result_name is not None:
        plt.savefig(result_name + "_summary.png")
        plt.close()
        # save the images
        with open(result_name + "_feats_and_stats.json", "w") as f:
            json.dump({"m1": m1,
                       "stat": stat,
                       "mode": mode,
                       "blocks_to_intervene": blocks_to_intervene,
                       }, f)
        img1.save(result_name + f"_{gsam_prompt2}_img1.png")
        img2.save(result_name + f"_{gsam_prompt1}_img2.png")
        result.save(result_name + ".png")
    else:
        plt.show()
