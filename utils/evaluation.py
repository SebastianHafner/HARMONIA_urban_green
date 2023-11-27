import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from utils import datasets, metrics


def model_evaluation(net, cfg, device: torch.device, run_type: str, epoch: float, max_samples: int = None):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.TrainDataset(cfg, run_type, no_augmentations=True)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=num_workers,
                                       shuffle=False, drop_last=False)

    stop_step = len(dataloader) if max_samples is None else max_samples
    for step, batch in enumerate(dataloader):
        if step == stop_step:
            break

        imgs = batch['x'].to(device)
        y_true = batch['y'].to(device)
        with torch.no_grad():
            y_pred = net(imgs)
        y_pred = torch.sigmoid(y_pred)
        measurer.add_sample(y_true.detach(), y_pred.detach())

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall

    # best f1 score for passed thresholds
    f1 = f1s.max()
    argmax_f1 = f1s.argmax()

    best_thresh = thresholds[argmax_f1]
    precision = precisions[argmax_f1]
    recall = recalls[argmax_f1]

    print(f'{f1.item():.3f}', flush=True)

    wandb.log(
        {f'{run_type} F1': f1,
               f'{run_type} threshold': best_thresh,
               f'{run_type} precision': precision,
               f'{run_type} recall': recall,
               'epoch': epoch,
        }
    )


def model_evaluation_change(net, cfg, device: torch.device, run_type: str, epoch: float, max_samples: int = None):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.TrainChangeDataset(cfg, run_type, no_augmentations=True)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=num_workers,
                                       shuffle=False, drop_last=False)

    stop_step = len(dataloader) if max_samples is None else max_samples
    for step, batch in enumerate(dataloader):
        if step == stop_step:
            break

        imgs_t1, imgs_t2 = batch['x_t1'].to(device), batch['x_t2'].to(device)
        y_true = batch['y'].to(device)
        with torch.no_grad():
            y_pred = net(imgs_t1, imgs_t2)
        y_pred = torch.sigmoid(y_pred)
        measurer.add_sample(y_true.detach(), y_pred.detach())

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall

    # best f1 score for passed thresholds
    f1 = f1s.max()
    argmax_f1 = f1s.argmax()

    best_thresh = thresholds[argmax_f1]
    precision = precisions[argmax_f1]
    recall = recalls[argmax_f1]

    print(f'{f1.item():.3f}', flush=True)

    wandb.log(
        {f'{run_type} F1': f1,
         f'{run_type} threshold': best_thresh,
         f'{run_type} precision': precision,
         f'{run_type} recall': recall,
         'epoch': epoch,
         }
    )
