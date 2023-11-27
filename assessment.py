import torch
from utils import networks, experiment_manager, datasets, parsers, metrics, geofiles
from tqdm import tqdm
from pathlib import Path
import numpy as np


def get_quantitative_results(cfg: experiment_manager.CfgNode, site: str, year: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading config and network
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    # loading dataset from config (requires inference.json)
    dataset = datasets.TilesInferenceDataset(cfg, site, year, include_label=True)

    y_probs, y_trues = [], []

    for i in tqdm(range(len(dataset))):
        patch = dataset.__getitem__(i)
        img = patch['x'].to(device)
        with torch.no_grad():
            logits = net(img.unsqueeze(0))
        y_prob = torch.sigmoid(logits)
        y_prob = y_prob.squeeze().cpu().numpy()
        y_prob = y_prob[dataset.patch_size:dataset.patch_size * 2, dataset.patch_size:dataset.patch_size * 2]
        y_probs.append(y_prob.flatten())

        y = patch['y'].squeeze().cpu().numpy()
        y_trues.append(y.flatten())

    y_probs, y_trues = np.concatenate(y_probs, axis=0), np.concatenate(y_trues, axis=0)
    data = {
        'f1_score': metrics.f1_score_from_prob(y_probs, y_trues),
        'precision': metrics.precision_from_prob(y_probs, y_trues),
        'recall': metrics.recall_from_prob(y_probs, y_trues),
        'iou': metrics.iou_from_prob(y_probs, y_trues),
        'kappa': metrics.kappa_from_prob(y_probs, y_trues),
    }
    data_file = Path(cfg.PATHS.OUTPUT) / 'assessment' / f'{cfg.NAME}_{site}_{year}_quantitative_results.json'
    geofiles.write_json(data_file, data)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    get_quantitative_results(cfg, args.site, int(args.year))