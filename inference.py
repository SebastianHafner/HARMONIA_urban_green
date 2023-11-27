from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils import geofiles, experiment_manager, networks, datasets, parsers


def run_inference(cfg: experiment_manager.CfgNode, site: str, year: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = datasets.TilesInferenceDataset(cfg, site, year)

    prob_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    for i in tqdm(range(len(dataset))):
        patch = dataset.__getitem__(i)
        img = patch['x'].to(device)
        with torch.no_grad():
            logits = net(img.unsqueeze(0))
        prob = torch.sigmoid(logits) * 100
        prob = prob.squeeze().cpu().numpy().astype('uint8')
        prob = np.clip(prob, 0, 100)
        center_prob = prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]

        i_start = patch['i']
        i_end = i_start + dataset.patch_size
        j_start = patch['j']
        j_end = j_start + dataset.patch_size
        prob_output[i_start:i_end, j_start:j_end, 0] = center_prob

    # config inference directory
    out_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    out_folder.mkdir(exist_ok=True)
    out_file = out_folder / f'pred_{site}{dataset.year}_{cfg.NAME}.tif'
    geofiles.write_tif(out_file, prob_output, transform, crs)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    run_inference(cfg, args.site, int(args.year))
