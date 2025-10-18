import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2
from functools import partial
import json
from gsam_utils import sam_mask, resize_mask
from typing import Literal
import inspect

class TimedHook:
    def __init__(self, hook_fn, total_steps, apply_at_steps=None):
        # check the signature of hook_fn
        if len(inspect.signature(hook_fn).parameters) != 4:
            raise ValueError(f"hook_fn must have 4 parameters: module, input, output, step_idx")
        self.hook_fn = hook_fn
        self.total_steps = total_steps
        self.apply_at_steps = apply_at_steps
        self.current_step = 0

    def identity(self, module, input, output):
        return output

    def __call__(self, module, input, output):
        if self.apply_at_steps is not None:
            if self.current_step in self.apply_at_steps:
                res = self.hook_fn(module, input, output, self.current_step)
                self.__increment()
                return res
            else:
                res = self.identity(module, input, output)
                self.__increment()
                return res

        res = self.identity(module, input, output)
        self.__increment()
        return res

    def __increment(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            self.current_step = 0


logger = logging.getLogger(__name__)

code_to_block = {
        "down.2.1": "unet.down_blocks.2.attentions.1",
        "up.0.1": "unet.up_blocks.0.attentions.1",
        "up.0.0": "unet.up_blocks.0.attentions.0",
        "mid.0": "unet.mid_block.attentions.0",
}

def add_featuremaps(sae, to_source_features, to_target_features, m1, fmaps, target_mask,  
                    maintain_spatial_info, module, input, output, step_idx):
    diff = output[0] - input[0]
    coefs = sae.encode(diff.permute(0, 2, 3, 1))
    mask = torch.zeros([1, fmaps.shape[1], fmaps.shape[2], sae.decoder.weight.shape[1]], device=input[0].device)
    # norm adjustment not needed because the columns have norm 1 already!
    norm_source = sae.decoder.weight[:, to_source_features].norm(dim=0).sum(dim=0)
    norm_target = sae.decoder.weight[:, to_target_features].norm(dim=0).sum(dim=0)
    if norm_target > 0:
        normadjustment = (norm_source/norm_target)
    else:
        normadjustment = 0
    target_update = normadjustment*m1*coefs[..., to_target_features] 
    target_update[:, ~target_mask] = 0
    if not maintain_spatial_info:
        target_update[:, target_mask] = target_update[:, target_mask].mean(dim=1, keepdim=True)
    mask[..., to_target_features] -= target_update
    mask[..., to_source_features] += fmaps[step_idx].unsqueeze(0).to(mask.device)
    to_add = mask.to(sae.decoder.weight.dtype) @ sae.decoder.weight.T
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


def add_activations(delta, module, input, output, step_idx):
    if isinstance(output, torch.Tensor):
        return output + delta[step_idx]
    else:  
        return (output[0] + delta[step_idx],)

def best_features_saeuron(source_feats, target_feats, k=10, aggregate_timesteps: Literal["first", "mean"] = "first"):
    assert len(source_feats.shape) == 3 and len(target_feats.shape) == 3, "source_feats and target_feats must be of shape (timesteps, spatial_locations, features)"
    if aggregate_timesteps == "first":
        source_feats = source_feats[0]
        target_feats = target_feats[0]
    elif aggregate_timesteps == "mean":
        source_feats = source_feats.mean(dim=0)
        target_feats = target_feats.mean(dim=0)
    else:
        raise ValueError(f"aggregate_timesteps {aggregate_timesteps} not supported")
    # now the first axis is the spatial locations, over which we just take the mean
    mean_source = source_feats.mean(dim=0)
    mean_target = target_feats.mean(dim=0)
    scores = mean_source/mean_source.sum() - mean_target/mean_target.sum()
    arg_sorted = np.argsort(scores.cpu().detach().numpy())
    return arg_sorted[::-1][:k].copy(), arg_sorted[:k].copy()

def best_features_neuron(source_feats, target_feats, k=10):
    mean_source = source_feats.mean(dim=0).mean(dim=0)
    mean_target = target_feats.mean(dim=0).mean(dim=0)
    scores = (mean_source/mean_source.sum() - mean_target/mean_target.sum()).abs()
    arg_sorted = np.argsort(scores.cpu().detach().numpy())
    return arg_sorted[::-1][:k].copy(), arg_sorted[:k].copy()

@torch.no_grad()
def get_features_per_block(cache1, cache2, mask1, mask2, 
                           k_transfer=None, blocks_to_intervene=None, saes=None,
                           aggregate_timesteps: Literal["first", "mean"] = "first"):
    to_source_features_dict = {}
    to_target_features_dict = {}
    source_feats_dict = {}
    target_feats_dict = {}
    source_dict = {}
    target_dict = {}
    for shortcut in blocks_to_intervene:
        block = code_to_block[shortcut]
        diff1 = cache1['output'][block][0] - cache1['input'][block][0]
        diff2 = cache2['output'][block][0] - cache2['input'][block][0]
        source = diff1[:, :, mask1]
        target = diff2[:, :, mask2]
        sae = saes[shortcut]
        source_feats = sae.encode(source.permute(0, 2, 1))
        target_feats = sae.encode(target.permute(0, 2, 1))
        to_source_features, to_target_features = best_features_saeuron(source_feats, target_feats, k=k_transfer, aggregate_timesteps=aggregate_timesteps)
        to_source_features_dict[shortcut] = to_source_features
        to_target_features_dict[shortcut] = to_target_features
        source_feats_dict[shortcut] = source_feats.detach().cpu()
        target_feats_dict[shortcut] = target_feats.detach().cpu()
        source_dict[shortcut] = source.detach().cpu()
        target_dict[shortcut] = target.detach().cpu()   
    return to_source_features_dict, to_target_features_dict, source_feats_dict, target_feats_dict, source_dict, target_dict
    
@torch.no_grad()
def get_features_all_blocks(cache1, cache2, mask1, mask2, k_transfer=None, blocks_to_intervene=None, saes=None,
                            normalize=False, aggregate_timesteps: Literal["first", "mean"] = "first"):
    source_feats_all = []
    target_feats_all = []
    to_source_feats_dict = {}
    to_target_feats_dict = {}
    source_feats_dict = {}
    target_feats_dict = {}
    source_dict = {}
    target_dict = {}
    block_sizes = {}
    for shortcut in blocks_to_intervene:
        block = code_to_block[shortcut]
        # cache1['output'][block] is of shape (batch, timesteps, channels, width, height) = (1, 4, 1280, 16, 16)
        # we assume batch size is 1 --> we index into the first element of the batch
        diff1 = cache1['output'][block][0] - cache1['input'][block][0]
        diff2 = cache2['output'][block][0] - cache2['input'][block][0]
        source = diff1[:, :, mask1]
        target = diff2[:, :, mask2]
        sae = saes[shortcut]
        source_feats = sae.encode(source.permute(0, 2, 1))
        target_feats = sae.encode(target.permute(0, 2, 1))
        if normalize:
            source_feats_all.append(source_feats/source_feats.mean(dim=1, keepdim=True).norm(dim=-1, keepdim=True))
            target_feats_all.append(target_feats/target_feats.mean(dim=1, keepdim=True).norm(dim=-1, keepdim=True))
        else:
            source_feats_all.append(source_feats)
            target_feats_all.append(target_feats)
        source_dict[shortcut] = source.detach().cpu()
        target_dict[shortcut] = target.detach().cpu()
        source_feats_dict[shortcut] = source_feats.detach().cpu() 
        target_feats_dict[shortcut] = target_feats.detach().cpu()
        block_sizes[shortcut] = source_feats.shape[-1]
    source_feats_all = torch.cat(source_feats_all, dim=-1)
    target_feats_all = torch.cat(target_feats_all, dim=-1)
    to_source_feats_all, to_target_feats_all = best_features_saeuron(source_feats_all, target_feats_all, k=k_transfer, aggregate_timesteps=aggregate_timesteps)
    start = 0
    for idx, shortcut in enumerate(blocks_to_intervene):
        to_source_feats = []
        to_target_feats = []
        for feat in to_source_feats_all:
            if feat >= start and feat < start+block_sizes[shortcut]:
                to_source_feats.append(feat-start)
        for feat in to_target_feats_all:
            if feat >= start and feat < start + block_sizes[shortcut]:
                to_target_feats.append(feat-start)
        to_source_feats_dict[shortcut] = np.asarray(to_source_feats)
        to_target_feats_dict[shortcut] = np.asarray(to_target_feats)
        start += block_sizes[shortcut]

    return to_source_feats_dict, to_target_feats_dict, source_feats_dict, target_feats_dict, source_dict, target_dict

def get_neuron_layer_name(block, lidx):
    return code_to_block[block] + f'.transformer_blocks.{lidx}.ff.net.0'

@torch.no_grad()
def get_neurons_all_blocks(cache1, cache2, mask1, mask2, k_transfer=None, blocks_to_intervene=None, saes=None,
                            normalize=True, aggregate_timesteps: Literal["first", "mean"] = "first"):
    to_source_neurons_dict = {}
    to_target_neurons_dict = {}
    source_dict = {}
    target_dict = {}
    block_sizes = {}
    neurons1_all = []
    neurons2_all = []
    for shortcut in blocks_to_intervene:
        for lidx in range(10):
            layer = get_neuron_layer_name(shortcut, lidx)
            neurons1 = cache1['output'][layer][0]
            neurons2 = cache2['output'][layer][0]
            neurons1 = neurons1[:, mask1.flatten(), :]
            neurons2 = neurons2[:, mask2.flatten(), :]
            if normalize:
                neurons1_all.append(neurons1/neurons1.norm(dim=1, keepdim=True))
                neurons2_all.append(neurons2/neurons2.norm(dim=1, keepdim=True))
            else:
                neurons1_all.append(neurons1)
                neurons2_all.append(neurons2)
            source_dict[layer] = neurons1.detach().cpu()
            target_dict[layer] = neurons2.detach().cpu()
            block_sizes[layer] = neurons1.shape[-1]
    neurons1_all = torch.cat(neurons1_all, dim=-1)
    neurons2_all = torch.cat(neurons2_all, dim=-1)
    to_source_neurons_all, to_target_neurons_all = best_features_saeuron(neurons1_all, neurons2_all, k=k_transfer, aggregate_timesteps=aggregate_timesteps)
    start = 0
    for idx, shortcut in enumerate(blocks_to_intervene):
        for lidx in range(10):
            layer = get_neuron_layer_name(shortcut, lidx)
            to_source_neurons = []
            to_target_neurons = []
            for nidx in to_source_neurons_all:
                if nidx >= start and nidx < start+block_sizes[layer]:
                    to_source_neurons.append(nidx-start)
            for nidx in to_target_neurons_all:
                if nidx >= start and nidx < start+block_sizes[layer]:
                    to_target_neurons.append(nidx-start)
            to_source_neurons_dict[layer] = np.asarray(to_source_neurons)
            to_target_neurons_dict[layer] = np.asarray(to_target_neurons)
            start += block_sizes[layer]
    return to_source_neurons_dict, to_target_neurons_dict, source_dict, target_dict

def setup_neuron_interventions(blocks_to_intervene, 
                               to_source_features_dict, 
                               to_target_features_dict, 
                               source_feats_dict, 
                               target_feats_dict, 
                               mask1, 
                               mask2, 
                               m1, 
                               stat, 
                               subtract_target_add_source, 
                               maintain_spatial_info,
                               n_steps,
                               device):
    interventions = {}
    for shortcut in blocks_to_intervene:
        for lidx in range(10):
            layer = get_neuron_layer_name(shortcut, lidx)
            # 1 x 39 x 5120
            # set up interventions 
            to_source_neurons = to_source_features_dict[layer]
            to_target_neurons = to_target_features_dict[layer]
            source_neurons = source_feats_dict[layer]
            target_neurons = target_feats_dict[layer]

            if stat == "quantile":
                dtype = source_neurons.dtype
                stat1_val = source_neurons.float().quantile(0.95, dim=1)
                stat1_val = stat1_val.to(dtype)[:, to_source_neurons]
                stat2_val = target_neurons.float().quantile(0.95, dim=1)
                stat2_val = stat2_val.to(dtype)[:, to_target_neurons] 
            elif stat == "mean":
                stat1_val = source_neurons.mean(dim=1)
                stat1_val = stat1_val[:, to_source_neurons]
                stat2_val = target_neurons.mean(dim=1)
                stat2_val = stat2_val[:, to_target_neurons]
            else:
                raise ValueError(f"stat {stat} not supported")
            # TODO: continue from here
            fmaps = torch.zeros((n_steps, 16 * 16, 5120), device=device, dtype=source_neurons.dtype)
            fmaps2 = torch.zeros((n_steps, 16 * 16, 5120), device=device, dtype=source_neurons.dtype)
            # example source_neurons shape torch.Size([4, 107, 5120])
            if subtract_target_add_source:        
                if maintain_spatial_info:
                    source_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    target_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    source_update[:, mask1.flatten(), :] = m1*source_neurons.to(device)
                    target_update[:, mask2.flatten(), :] = m1*target_neurons.to(device)
                    fmaps[..., to_source_neurons] += source_update[..., to_source_neurons]
                    fmaps[..., to_target_neurons] -= target_update[..., to_target_neurons]
                else:
                    source_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    target_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    source_update[..., to_source_neurons] = m1*stat1_val.unsqueeze(1).to(device)
                    target_update[..., to_target_neurons] = m1*stat2_val.unsqueeze(1).to(device)
                    source_update[:, ~mask1.flatten(), :] = 0
                    target_update[:, ~mask2.flatten(), :] = 0
                    fmaps[..., to_source_neurons] += source_update[..., to_source_neurons]
                    fmaps[..., to_target_neurons] -= target_update[..., to_target_neurons]
            else:
                if maintain_spatial_info:
                    source_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    source_update[..., to_source_neurons] = m1*stat1_val.unsqueeze(1).to(device)
                    source_update[:, ~mask2.flatten(), :] = 0
                    target_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    target_update[:, mask2.flatten(), :] = m1*target_neurons.to(device)
                    fmaps[..., to_source_neurons] += source_update[..., to_source_neurons]
                    fmaps[..., to_target_neurons] -= target_update[..., to_target_neurons]
                else:
                    source_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    target_update = torch.zeros_like(fmaps, device=device, dtype=source_neurons.dtype)
                    source_update[..., to_source_neurons] = m1*stat1_val.unsqueeze(1).to(device)
                    target_update[..., to_target_neurons] = m1*stat2_val.unsqueeze(1).to(device)
                    source_update[:, ~mask2.flatten(), :] = 0
                    target_update[:, ~mask2.flatten(), :] = 0
                    fmaps[..., to_source_neurons] += source_update[..., to_source_neurons]
                    fmaps[..., to_target_neurons] -= target_update[..., to_target_neurons]
                    
            f = partial(add_activations, fmaps)
            hook = TimedHook(f, n_steps, apply_at_steps=list(range(n_steps)))
            interventions[layer] = hook
    return interventions

def setup_activation_interventions(blocks_to_intervene, 
                                   to_source_features_dict, 
                                   to_target_features_dict, 
                                   source_feats_dict, 
                                   target_feats_dict, 
                                   source_dict,
                                   target_dict,
                                   mask1, 
                                   mask2, 
                                   m1, 
                                   stat, 
                                   subtract_target_add_source, 
                                   maintain_spatial_info, 
                                   mode,
                                   saes,
                                   n_steps,
                                   device):
    interventions = {}
    for shortcut in blocks_to_intervene:
        # 1 x 39 x 5120
        # set up interventions 
        to_source_features = to_source_features_dict[shortcut]
        to_target_features = to_target_features_dict[shortcut]
        source_feats = source_feats_dict[shortcut]
        target_feats = target_feats_dict[shortcut]
        source = source_dict[shortcut]
        target = target_dict[shortcut]
        sae = saes[shortcut]
        block = code_to_block[shortcut]
        logger.debug("Selecting best features...")

        # source_feates example shape torch.Size([4, 135, 5120])
        if stat == "max":
            stat1_val = source_feats.max(dim=1)[0][:, to_source_features]
            stat2_val = target_feats.max(dim=1)[0][:, to_target_features]
        elif stat == "mean":
            stat1_val = source_feats.mean(dim=1)[:, to_source_features]
            stat2_val = target_feats.mean(dim=1)[:, to_target_features]
        elif stat == "quantile":
            logger.debug(f"source_feats shape: {source_feats.shape}")
            dtype = source_feats.dtype
            stat1_val = source_feats.float().quantile(0.95, dim=1)
            stat1_val = stat1_val.to(dtype)[:, to_source_features]
            stat2_val = target_feats.float().quantile(0.95, dim=1)
            stat2_val = stat2_val.to(dtype)[:, to_target_features]
        else:
            ValueError(f"stat1 {stat} not recognized. Choose from: max, mean")
        logger.debug("Running SDXL with feature injection...")

        if mode == "steering":
            delta = torch.zeros((n_steps, 1280, 16, 16), dtype=source.dtype, device=device)
            if subtract_target_add_source:
                if maintain_spatial_info:
                    delta[:, :, mask1] += m1*source.to(delta.device)
                    delta[:, :, mask2] -= m1*target.to(delta.device)
                else:
                    delta[:, :, mask1] += m1*source.mean(dim=2, keepdim=True).to(delta.device)
                    delta[:, :, mask2] -= m1*target.mean(dim=2, keepdim=True).to(delta.device)
            else:
                if maintain_spatial_info:
                    delta[:, :, mask2] += m1*source.mean(dim=2, keepdim=True).to(delta.device)
                    delta[:, :, mask2] -= m1*target.to(delta.device)
                else:
                    delta[:, :, mask2] += m1*source.mean(dim=2, keepdim=True).to(delta.device)
                    delta[:, :, mask2] -= m1*target.mean(dim=2, keepdim=True).to(delta.device)
            f = partial(add_activations, delta)
            hook = TimedHook(f, n_steps, apply_at_steps=list(range(n_steps)))
            interventions[block] = hook
        elif mode == "sae":
            delta = torch.zeros((n_steps, 1280, 16, 16), device=device, dtype=source.dtype)
            if subtract_target_add_source:
                # sae.decoder.weight example shape torch.Size([1280, 5120]) (this is from an expansion factor 4)
                # source_feats example shape torch.Size([4, 135, 5120]) 
                if maintain_spatial_info:
                    source_update = source_feats[..., to_source_features].to(device) @ sae.decoder.weight.T[to_source_features]
                    source_update = m1 * source_update.permute(0, 2, 1)
                    target_update = target_feats[..., to_target_features].to(device) @ sae.decoder.weight.T[to_target_features]
                    target_update = m1 * target_update.permute(0, 2, 1)
                    delta[:, :, mask1] += source_update
                    delta[:, :, mask2] -= target_update
                else:
                    source_update = stat1_val.unsqueeze(1).to(device) @ sae.decoder.weight.T[to_source_features]
                    source_update = m1 * source_update.permute(0, 2, 1)
                    target_update = stat2_val.unsqueeze(1).to(device) @ sae.decoder.weight.T[to_target_features]
                    target_update = m1 * target_update.permute(0, 2, 1)
                    delta[:, :, mask1] += source_update
                    delta[:, :, mask2] -= target_update
            else:
                if maintain_spatial_info:
                    source_update = stat1_val.unsqueeze(1).to(device) @ sae.decoder.weight.T[to_source_features]
                    source_update = m1 * source_update.permute(0, 2, 1)
                    target_update = target_feats[..., to_target_features].to(device) @ sae.decoder.weight.T[to_target_features]
                    target_update = m1 * target_update.permute(0, 2, 1)
                    delta[:, :, mask2] += source_update
                    delta[:, :, mask2] -= target_update
                else:
                    source_update = stat1_val.unsqueeze(1).to(device) @ sae.decoder.weight.T[to_source_features]
                    source_update = m1 * source_update.permute(0, 2, 1)
                    target_update = stat2_val.unsqueeze(1).to(device) @ sae.decoder.weight.T[to_target_features]
                    target_update = m1 * target_update.permute(0, 2, 1)
                    delta[:, :, mask2] += source_update
                    delta[:, :, mask2] -= target_update

            f = partial(add_activations, delta)
            hook = TimedHook(f, n_steps, apply_at_steps=list(range(n_steps)))
            interventions[block] = hook
        elif mode == "sae_adaptive":
            fmaps = torch.zeros((n_steps, 16, 16, len(to_source_features)), device=device)
            if subtract_target_add_source:       
                if maintain_spatial_info:
                    fmaps[:, mask1] += m1*source_feats[..., to_source_features].to(fmaps.device)
                else: 
                    fmaps[:, mask1] += (m1*stat1_val).unsqueeze(1).to(fmaps.device)
            else: 
                fmaps[:, mask2] += (m1*stat1_val).unsqueeze(1).to(fmaps.device)
            f = partial(add_featuremaps, sae, to_source_features, to_target_features, m1, fmaps, mask2, maintain_spatial_info)
            hook = TimedHook(f, n_steps, apply_at_steps=list(range(n_steps)))
            interventions[block] = hook
        else:
            ValueError(f"Mode {mode} not recognized. Choose from: patch_max, patch_mean, sae, neurons, steer")
    return interventions


def run_feature_transport(prompt1, prompt2, gsam_prompt1, gsam_prompt2, pipe, grounding_model, sam2_predictor, saes,
         blocks_to_intervene=["down.2.1", "up.0.1", "up.0.0", "mid.0"],
         n_steps=1, m1=1., k_transfer=10, 
         stat: Literal["max", "mean", "quantile"] = "quantile", 
         mode: Literal["steering", "sae", "neurons"] = "sae",  
         combine_blocks=True, use_source_mask_in_both=False, subtract_target_add_source=False,
         maintain_spatial_info=False, 
         aggregate_timesteps: Literal["first", "mean"] = "first", 
         verbose=False,
         BOX_THRESHOLD=0.25, TEXT_THRESHOLD=0.25,
         result_name=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)    

    blocks = [code_to_block[shortcut] for shortcut in blocks_to_intervene]
    if mode == "neurons":
        neuron_blocks = []
        for shortcut in blocks_to_intervene:
            for idx in range(10):
                neuron_blocks.append(get_neuron_layer_name(shortcut, idx))
        blocks = neuron_blocks

    logger.debug("Generating images and caching activations...")
    seed = 42
    base_imgs1, cache1 = pipe.run_with_cache(
        prompt1,
        positions_to_cache=blocks,
        num_inference_steps=n_steps,
        guidance_scale=0.0,
        generator=torch.Generator(device='cpu').manual_seed(seed),
        save_input=True,
    )
    base_imgs2, cache2 = pipe.run_with_cache(
        prompt2,
        positions_to_cache=blocks,
        num_inference_steps=n_steps,
        guidance_scale=0.0,
        generator=torch.Generator(device='cpu').manual_seed(seed),
        save_input=True,
    )
    img1 = base_imgs1[0][0]
    img2 = base_imgs2[0][0]

    logger.debug("Running Grounded SAM on generated images...")
    if gsam_prompt1 == "#everything":
        mask1 = np.ones((16, 16), dtype=bool)
    else:
        if "background" in gsam_prompt1:
            detections1, labels1, annotated_frame1 = sam_mask(img1, "foreground", sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask).astype(np.float32) for bigmask in detections1.mask]
            mask1 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
            mask1 = np.logical_not(mask1)
            if verbose:
                plt.imshow(mask1)
                plt.show()
        elif "~" in gsam_prompt1:
            detections1, labels1, annotated_frame1 = sam_mask(img1, gsam_prompt1.replace("~", ""), sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask).astype(np.float32) for bigmask in detections1.mask]
            mask1 = ~np.stack(masks, axis=0).sum(axis=0).astype(bool)
        else:
            detections1, labels1, annotated_frame1 = sam_mask(img1, gsam_prompt1, sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask).astype(np.float32) for bigmask in detections1.mask]
            mask1 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
    if gsam_prompt2 == "#everything":
        mask2 = np.ones((16, 16), dtype=bool)
    else:
        if "background" in gsam_prompt2:
            detections2, labels2, annotated_frame2 = sam_mask(img2, "foreground", sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask).astype(np.float32) for bigmask in detections2.mask]
            mask2 = np.stack(masks, axis=0).sum(axis=0).astype(bool)
            mask2 = np.logical_not(mask2)
            if verbose:
                plt.imshow(mask2)
                plt.show()
        elif "~" in gsam_prompt2:
            detections2, labels2, annotated_frame2 = sam_mask(img2, gsam_prompt2.replace("~", ""), sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask).astype(np.float32) for bigmask in detections2.mask]
            mask2 = ~np.stack(masks, axis=0).sum(axis=0).astype(bool)
        else:
            detections2, labels2, annotated_frame2 = sam_mask(img2, gsam_prompt2, sam2_predictor, grounding_model, BOX_THRESHOLD, TEXT_THRESHOLD)
            masks = [resize_mask(bigmask).astype(np.float32) for bigmask in detections2.mask]
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
    
    if mode == "neurons":
        to_source_features_dict, to_target_features_dict, source_feats_dict, target_feats_dict = \
            get_neurons_all_blocks(cache1, cache2, mask1, mask2, 
                                   k_transfer=k_transfer, 
                                   blocks_to_intervene=blocks_to_intervene,
                                   aggregate_timesteps=aggregate_timesteps)
    elif combine_blocks:
        to_source_features_dict, to_target_features_dict, source_feats_dict, target_feats_dict, source_dict, target_dict = \
            get_features_all_blocks(cache1, cache2, mask1, mask2,
                                   k_transfer=k_transfer, blocks_to_intervene=blocks_to_intervene, saes=saes,
                                   aggregate_timesteps=aggregate_timesteps)
    else:
        to_source_features_dict, to_target_features_dict, source_feats_dict, target_feats_dict, source_dict, target_dict = \
            get_features_per_block(cache1, cache2, mask1, mask2,
                                   k_transfer=k_transfer, blocks_to_intervene=blocks_to_intervene, saes=saes,
                                   aggregate_timesteps=aggregate_timesteps)

    if mode != "neurons":
        interventions = setup_activation_interventions(blocks_to_intervene, 
                                                       to_source_features_dict, 
                                                       to_target_features_dict, 
                                                       source_feats_dict, 
                                                       target_feats_dict, 
                                                       source_dict, 
                                                       target_dict, 
                                                       mask1, 
                                                       mask2, 
                                                       m1, 
                                                       stat, 
                                                       subtract_target_add_source, 
                                                       maintain_spatial_info, 
                                                       mode,
                                                       saes,
                                                       n_steps, 
                                                       device)
    else:
        interventions = setup_neuron_interventions(blocks_to_intervene, 
                                                   to_source_features_dict, 
                                                   to_target_features_dict, 
                                                   source_feats_dict, 
                                                   target_feats_dict, 
                                                   mask1, 
                                                   mask2, 
                                                   m1, 
                                                   stat, 
                                                   subtract_target_add_source, 
                                                   maintain_spatial_info, 
                                                   n_steps,
                                                   device)
                

    result = pipe.run_with_hooks(
        prompt2,
        position_hook_dict=interventions,
        num_inference_steps=n_steps,
        guidance_scale=0.0,
        generator=torch.Generator(device='cpu').manual_seed(seed)
    ).images[0]

    # make a result figure that shows the images with masks and the intervened image
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image 1 with mask from prompt 1
    if "~" not in gsam_prompt1 and "background" not in gsam_prompt1 and gsam_prompt1 != "#everything":
        axs[0].imshow(annotated_frame1)
    else:
        if "background" in gsam_prompt1 or "~" in gsam_prompt1:
            # upsample mask by 32
            mask1 = cv2.resize(mask1.astype(np.float32), (16*32, 16*32), interpolation=cv2.INTER_NEAREST)
            # Plot the base image first
            axs[0].imshow(img1)
            # Then overlay the mask with higher alpha for visibility
            axs[0].imshow(mask1, alpha=0.5)
        else:
            axs[0].imshow(img1)
    axs[0].set_title(f"{prompt1}", fontsize=16)
    axs[0].axis('off')
    
    # Image 2 with mask from prompt 2
    if "~" not in gsam_prompt2 and "background" not in gsam_prompt2 and gsam_prompt2 != "#everything":
        axs[1].imshow(annotated_frame2)
    else:
        if "background" in gsam_prompt2 or "~" in gsam_prompt2:
            # upsample mask by 32
            mask2 = cv2.resize(mask2.astype(np.float32), (16*32, 16*32), interpolation=cv2.INTER_NEAREST)
            # Plot the base image first
            axs[1].imshow(img2)
            # Then overlay the mask with higher alpha for visibility
            axs[1].imshow(mask2, alpha=0.5)
        else:
            axs[1].imshow(img2)
    axs[1].set_title(f"{prompt2}", fontsize=16)
    axs[1].axis('off')
    
    # Intervened result image
    axs[2].imshow(result)
    axs[2].axis('off')

    if result_name is not None:
        plt.tight_layout()
        plt.savefig(result_name + "_summary.png", bbox_inches='tight')
        plt.close()
        # save the images
        with open(result_name + "_feats_and_stats.json", "w") as f:
            json.dump({"to_source_features": {k:v.tolist() for k,v in to_source_features_dict.items()}, 
                       "to_target_features": {k:v.tolist() for k,v in to_target_features_dict.items()},
                       "m1": m1,
                       "k_transfer": k_transfer,
                       "stat": stat,
                       "mode": mode,
                       "combine_blocks": combine_blocks,
                       "blocks_to_intervene": blocks_to_intervene,
                       }, f)
        img1.save(result_name + f"_{gsam_prompt2}_img1.png")
        img2.save(result_name + f"_{gsam_prompt1}_img2.png")
        result.save(result_name + ".png")
    else:
        print(to_source_features_dict)
        print(to_target_features_dict)
        plt.show()