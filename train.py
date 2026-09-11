import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import accelerate
import gc
import os
import time
import torch
import torch.distributed as dist

from datetime import datetime

from accelerate import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ScpAudioDataset
from models.model_Sphere_VAE import build_model
from losses import MultiScaleMelSpectrogramLoss
from losses import generator_loss, feature_loss, discriminator_loss
from discriminators import MultiScaleSTFTDiscriminator
from utils import utils
from torch.utils.data import DataLoader, DistributedSampler


from transformers import get_cosine_schedule_with_warmup
import math

torch.backends.cudnn.benchmark = True
global_step = 0
device = None
use_cuda = torch.cuda.is_available()
      
def collate_fn(batch):
    """
    过滤 None 项，堆叠 wav
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:

        return None
    
    wavs = [b["wav"] for b in batch]
    utts = [b["utt"] for b in batch]
    texts = [b["text"] for b in batch]

    wavs = torch.stack(wavs)

    return {
        "wav": wavs,
        "utt": utts,
        "text": texts
    }


def create_dataloader(scp_path, batch_size, num_workers, sr, segment_size, seed_value=42):
    dataset = ScpAudioDataset(scp_path, sr, segment_size, seed_value=seed_value)


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=False
    )

    return loader

class WarmupCosineScheduler:
    """封装warmup + cosine，使用transformers库实现，支持checkpoint"""
    def __init__(self, optimizer, total_steps, warmup_steps=2000):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=1
        )
    
    def step(self):
        self.scheduler.step()
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
        

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def main():
    """Assume Single Node Multi GPUs Training Only"""

    
    hps = utils.get_hparams()
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(

        gradient_accumulation_steps=hps.train.gradient_accumulation_steps,

    )
    accelerator.wait_for_everyone()
    run(accelerator, hps)
    
def run(accelerator: accelerate.Accelerator, hps: utils.HParams):
    global global_step, device
    if accelerator.is_main_process:
        logger = utils.get_logger(hps.train.save_dir)
        logger.info(hps.train)
        logger.info(hps.data)
        logger.info(hps.model)
        utils.check_git_hash(hps.train.save_dir)
        writer = SummaryWriter(log_dir=hps.train.save_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.save_dir, "eval"))
    
    torch.manual_seed(hps.data.seed)
    net_g = build_model(
        d_model=hps.model.d_model,
        seanet_kwargs=hps.model.seanet,
        transformer_kwargs=hps.model.transformer,
        latent_dimension=hps.model.latent_dimension,
    )
    net_d = MultiScaleSTFTDiscriminator(
       **hps.model.msstftd
    )
    msspec = MultiScaleMelSpectrogramLoss(
       **hps.model.msspec
    )
    
    params_g = []
    for name, param in net_g.named_parameters():
        if 'encoder_transformer' in name:
            params_g.append({
                'params': param,
                'weight_decay': 5e-2,
                'name': name
            })
        elif 'decoder_transformer' in name:
            params_g.append({
                'params': param,
                'weight_decay': 5e-2,
                'name': name
            })
        else:
            params_g.append({
                'params': param,
                'weight_decay': 0.0,
                'name': name
            })
            

    optim_g = torch.optim.AdamW(
        params_g,
        lr=hps.train.learning_rate,
        betas=(0.5, 0.9),
        eps=1e-8
    )

    optim_d = torch.optim.AdamW(
        [{'params': net_d.parameters()}],
        lr=hps.train.learning_rate,
        betas=(0.5, 0.9),
        eps=1e-8,
        weight_decay=0.0 
    )
    

    scheduler_g = WarmupCosineScheduler(optim_g, hps.train.training_steps, warmup_steps=200)
    scheduler_d = WarmupCosineScheduler(optim_d, hps.train.training_steps, warmup_steps=200)
    
    try:
        net_g, optim_g, scheduler_g, _, epoch_str, step = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"),
            net_g,
            optim_g,
            scheduler_g,
        )
        net_d, optim_d, scheduler_d, _, epoch_str, step = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"),
            net_d,
            optim_d,
            scheduler_d,
        )
        global_step = step
    except Exception:
        import traceback

        traceback.print_exc()
        epoch_str = 1
        global_step = 0

        # Fall back to model weights when optimizer/scheduler state is absent
        # or incompatible. A generator-only checkpoint is sufficient to start
        # fine-tuning; the discriminator remains freshly initialized if no D
        # checkpoint is available.
        try:
            net_g = utils.load_checkpoint_weight_only(
                utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"),
                net_g,
            )
        except Exception:
            traceback.print_exc()

        try:
            net_d = utils.load_checkpoint_weight_only(
                utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"),
                net_d,
            )
        except Exception:
            print("No compatible discriminator checkpoint; using a fresh discriminator.")


    train_loader = create_dataloader(
        hps.data.train_scp_path,
        batch_size=hps.data.batch_size,
        num_workers=hps.data.num_workers,
        sr=hps.data.sample_rate,
        segment_size=hps.data.segment_size,
        seed_value=hps.data.seed
    )


    (
        net_g, 
        optim_g, 
        scheduler_g, 
        net_d, 
        optim_d, 
        scheduler_d, 
        train_loader,
        msspec,
    ) = accelerator.prepare(
        net_g, 
        optim_g, 
        scheduler_g, 
        net_d, 
        optim_d, 
        scheduler_d, 
        train_loader,
        msspec,
    )

    for epoch in range(epoch_str, hps.train.epochs + 1):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{hps.train.epochs}"):
            if batch == None:
                continue

            if accelerator.is_main_process:
                train_and_evaluate(
                    accelerator, 
                    epoch, 
                    hps, 
                    [net_g, net_d, msspec], 
                    [optim_g, optim_d], 
                    [scheduler_g, scheduler_d], 
                    batch, 
                    logger, 
                    [writer, writer_eval]
                )
            else:
                train_and_evaluate(
                    accelerator, 
                    epoch, 
                    hps, 
                    [net_g, net_d, msspec], 
                    [optim_g, optim_d], 
                    [scheduler_g, scheduler_d], 
                    batch, 
                    None, 
                    None
                )
            scheduler_g.step()
            scheduler_d.step()
            if global_step >= 10000000000:
                logger.info('End training at step 100w')
                break

def train_and_evaluate(
        accelerator: accelerate.Accelerator, 
        epoch, 
        hps, 
        nets, 
        optims, 
        schedulers, 
        item, 
        logger, 
        writers
    ):
    net_g, net_d, msspec = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    
    global global_step


    wavs, keys = item['wav'], item['utt']
    if writers is not None:
        writer, writer_eval = writers

    net_g.train()
    net_d.train()

    wavs = wavs.unsqueeze(1).to(accelerator.device)
    result = net_g(wavs)
    loss_kl = result.metrics["kl_loss"]


    y_d_hat_r, y_d_hat_g, _, _ = net_d(wavs, result.recon.detach())
    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    
    loss_disc_all = loss_disc


    optim_d.zero_grad()
    accelerator.backward(loss_disc_all)


    grad_norm_d = clip_grad_value_(net_d.parameters(), 1.0)
    optim_d.step()

    wavs = wavs.to(result.recon.device)

    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wavs, result.recon)


    loss_msspec = msspec(result.recon, wavs) * 15

    
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)

    loss_gen_all = loss_gen + 1.5*loss_fm + loss_msspec + 0.01 * loss_kl

    optim_g.zero_grad()


    accelerator.backward(loss_gen_all)

    grad_norm_g = clip_grad_value_(net_g.parameters(), 1.0)
    optim_g.step()


    if accelerator.is_main_process:
        if global_step % hps.train.log_interval == 0:

            logger.info('====> Epoch: {}'.format(epoch))
            logger.info([
                global_step, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                loss_msspec.item(),
                loss_fm.item(),
                loss_gen.item(),
                loss_kl.item()
            ])
            lr = optim_g.param_groups[0]['lr']


            scalar_dict = {
                "loss/total": loss_gen_all, 
                "loss/msspec": loss_msspec,
                "loss/fm": loss_fm,
                "loss/gen": loss_gen,
                "loss/disc": loss_disc,
                "loss/kl": loss_kl,
                "learning_rate": lr,
                "grad_norm_g": grad_norm_g,
                "grad_norm_d": grad_norm_d
            }
            scalar_dict.update(result.metrics)  
            utils.summarize(
                writer=writer,
                global_step=global_step, 
                scalars=scalar_dict
            )

        if global_step % hps.train.eval_interval == 0:
            logger.info(['All training params(G): ', utils.count_parameters(net_g), ' M'])

        if global_step % hps.train.eval_interval == 0:
            utils.save_checkpoint(net_g, optim_g, scheduler_g, hps.train.learning_rate, epoch, global_step, os.path.join(hps.train.save_dir, "G_{}.pth".format(global_step)))
            utils.save_checkpoint(net_d, optim_d, scheduler_d, hps.train.learning_rate, epoch, global_step, os.path.join(hps.train.save_dir, "D_{}.pth".format(global_step)))
            net_g.train()
    global_step += 1    

 
if __name__ == "__main__":
    main()
