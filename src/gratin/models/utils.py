from collections import defaultdict
import numpy as np
from tqdm.notebook import tqdm
import torch


def get_predictions_of_dl(model, dl, latent_samples=0):

    outputs = defaultdict(lambda: [])
    info = defaultdict(lambda: [])
    if latent_samples > 0:
        latent = []
    if len(dl) > 300:
        wrapper = tqdm
    else:
        wrapper = lambda x: x
    for batch in wrapper(dl):
        batch = batch.to(model.device)
        with torch.no_grad():
            out, h = model(batch)
        for key in out:
            outputs[key].append(out[key].detach().cpu().numpy())
        info["alpha"].append(batch.alpha[:, 0].detach().cpu().numpy())
        info["model"].append(batch.model[:, 0].detach().cpu().numpy())
        info["length"].append(batch.length[:, 0].detach().cpu().numpy())
        info["noise"].append(batch.noise[:, 0].detach().cpu().numpy())
        info["log_tau"].append(batch.log_tau[:, 0].detach().cpu().numpy())
        info["log_diffusion"].append(batch.log_diffusion[:, 0].detach().cpu().numpy())
        if latent_samples > 0:
            if len(latent) * h.shape[0] <= latent_samples:
                latent.append(h.detach().cpu().numpy())
            else:
                break

    for key in outputs:
        outputs[key] = np.concatenate(outputs[key], axis=0)
        if outputs[key].shape[1] == 1:
            outputs[key] = outputs[key][:, 0]

    for key in info:
        info[key] = np.concatenate(info[key], axis=0)

    if latent_samples > 0:
        h = np.concatenate(latent, axis=0)
        for key in info:
            info[key] = info[key][: h.shape[0]]
        for key in outputs:
            outputs[key] = outputs[key][: h.shape[0]]
        return outputs, info, h

    return outputs, info
