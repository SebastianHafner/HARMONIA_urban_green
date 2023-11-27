import sys
from pathlib import Path
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg):
    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.TrainDataset(cfg, 'train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0
    n_pixels = n_pos_pixels = 0
    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)

            y_pred = net(x)

            loss = criterion(y_pred, y_gts)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            n_pixels += y_gts.numel()
            n_pos_pixels += torch.sum(y_gts)

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, device, 'train', epoch_float, max_samples=1_000)
                evaluation.model_evaluation(net, cfg, device, 'val', epoch_float)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'pos_pixel_percentage': float(n_pos_pixels / n_pixels * 100),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []

            # end of batch

        assert(epoch == epoch_float)
        evaluation.model_evaluation(net, cfg, device, 'train', epoch_float, global_step)
        evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        if not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, cfg)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project=args.project,
        tags=['run', 'urban', 'green', 'segmentation', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
