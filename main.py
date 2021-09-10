#!/usr/bin/env python3

# modify from
# https://github.com/pytorch/elastic/blob/4d07d4eeb08449ca1ec6afed2f97da97c6809621/examples/imagenet/main.py

import argparse
import io
import os
import shutil
import time
from contextlib import contextmanager
from datetime import timedelta
from datetime import datetime
from typing import List, Tuple

import numpy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader

# my add
from transformer_net import TransformerNet
import glob
import pprint
from custom_dataset import StyleIdxDataset
from string import Template
from pathlib import Path
import utils
from vgg import VGG
from collections import defaultdict

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser(description="PyTorch Elastic ImageNet Training")
parser.add_argument(
  "--data",
  metavar="DIR",
  default="val2017",
  help="path to dataset"
)

parser.add_argument(
  "--epochs",
  default=50,
  type=int,
  metavar="N", help="number of total epochs to run"
)

parser.add_argument(
  "-j",
  "--workers",
  default=0,
  type=int,
  metavar="N",
  help="number of data loading workers",
)

parser.add_argument(
  "-b",
  "--batch-size",
  default=16,
  type=int,
  metavar="N",
  help="mini-batch size (default: 16), per worker (GPU)",
)



parser.add_argument(
  "--lr",
  "--learning-rate",
  default=1e-3,
  type=float,
  metavar="LR",
  help="initial learning rate",
  dest="lr",
)

parser.add_argument(
  "--wd",
  "--weight-decay",
  default=1e-2,
  type=float,
  metavar="W",
  help="weight decay (default: 1e-4)",
  dest="weight_decay",
)

parser.add_argument(
  "--content-weight",
  default=1e5,
  type=float,
  metavar="W",
  help="content weight (default: 1e5)",
)

parser.add_argument(
  "--style-weight",
  default=1e10,
  type=float,
  metavar="W",
  help="style weight (default: 1e10)",
)

parser.add_argument(
  "--content-layers",
  nargs='+',
  default=["relu2_2"],
  help="content-layers (default: [\"relu2_2\"])",
)

parser.add_argument(
  "--style-layers",
  nargs='+',
  default=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
  help="style-layers"
       "(default: ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])",
)

parser.add_argument(
  "-p",
  "--print-freq",
  default=50,
  type=int,
  metavar="N",
  help="print every n batch (default: 50)",
)

parser.add_argument(
  "--style-size",
  type=int,
  help="the resized shape of style_image (default: None)",
)

parser.add_argument(
  "--dist-backend",
  default="gloo",
  choices=["nccl", "gloo"],
  type=str,
  help="distributed backend",
)

parser.add_argument(
  "--checkpoint-file",
  default="${FOLDER}/checkpoint_E_${EPOCH}.pth",
  type=str,
  help="checkpoint file path, to load and save to",
)

parser.add_argument(
  "--output-folder",
  default="",
  type=str,
  help="output folder, default : \"\""
)

parser.add_argument(
  "--style-paths-template",
  default="./style/*.jpg",
  type=str,
)

parser.add_argument(
  "--seed",
  default=228922,
  type=int,
)

def main():
  args = parser.parse_args()
  
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  
  device_id = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(device_id)
  
  print(f"=> set cuda device = {device_id}")
  dist.init_process_group(
    backend=args.dist_backend,
    init_method="env://",
    timeout=timedelta(seconds=10)
  )
  
  style_images_path = glob.glob(args.style_paths_template)
  n_style = len(style_images_path)
  
  model, optimizer = initalize_model(
    n_style, args.lr, args.batch_size, args.workers, args.weight_decay, device_id
  )
  criterion = initalize_criterion(
    args.content_layers,
    args.style_layers,
    args.content_weight,
    args.style_weight,
    style_images_path,
    args.style_size,
    device_id,
  )
  
  train_loader, style_idx_loader = initialize_data_loader(
    args.data, n_style, args.batch_size, args.workers
  )
  
  args.output_folder = args.output_folder or datetime.now().strftime("result_%Y-%m-%d-%H-%M")
  if device_id == 0:
    Path(args.output_folder).mkdir(parents=True, exist_ok=False)
  
  args.checkpoint_file = Template(args.checkpoint_file).safe_substitute(FOLDER=args.output_folder)
  
  state = load_checkpoint(
    args.checkpoint_file,
    device_id, model, optimizer
  )

  start_epoch = state.epoch
  print(f"=> start_epoch: {start_epoch}")
  print_freq = args.print_freq
  
  
  def set_random_seed(*loaders, epoch=0):
    for nth_loader, l in enumerate(loaders):
      l.batch_sampler.sampler.set_epoch(epoch+nth_loader)
    
  for epoch in range(start_epoch, args.epochs):
    state.epoch = epoch
    set_random_seed(train_loader, style_idx_loader, epoch=epoch)

    # train for one epoch
    train(train_loader, style_idx_loader, model, criterion, optimizer, epoch, device_id, print_freq)

    if device_id == 0:
      model.eval()
      save_checkpoint(state, Template(args.checkpoint_file).safe_substitute(EPOCH=str(epoch).zfill(5)))

class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model, optimizer):
        self.epoch = -1
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)

  
def initalize_model(n_style, lr, batch_size, workers, weight_decay, device_id):
  model = TransformerNet(n_style)
  model.cuda(device_id)
  cudnn.benchmark = True
  
  model = DistributedDataParallel(model, device_ids=[device_id])
  optimizer = Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
  )
  
  return model, optimizer

def training_image_transforms(big_size=512, image_size=256):
  return transforms.Compose([
    transforms.Resize(big_size),
    transforms.RandomCrop(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
  ])

def initalize_criterion(
  content_layers,
  style_layers,
  content_weight,
  style_weight,
  style_image_paths,
  style_size,
  device_id,
  loss_fun = torch.nn.MSELoss(),
):
  vgg = VGG(requires_grad=False).to(device_id)
  feature_layers = list(set(
    content_layers+style_layers
  ))
  
  style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x:x.mul(255))
  ])
  
  style_imgs = [
    utils.load_image(path, style_size)
    for path in style_image_paths
  ]
  style_imgs = [
    style_transform(img)
    for img in style_imgs
  ]
  
  gram_style= defaultdict(list)
  for style in style_imgs:
    for k, v in vgg(utils.normalize_batch(style.unsqueeze(0).to(device_id)), style_layers).items():
      gram_style[k].append(utils.gram_matrix(v))

  gram_style = {
    layer_name : torch.cat(gram) for layer_name, gram in gram_style.items()
  }
  
  def criterion(stylized_result, images, style_idx):
    x = images = utils.normalize_batch(images)
    y = stylized_result = utils.normalize_batch(stylized_result)
    
    features_y = vgg(y, feature_layers)
    features_x = vgg(x, content_layers)
    
    content_loss = 0
    for layer_name in content_layers:
      content_loss += loss_fun(features_y[layer_name], features_x[layer_name])
    content_loss *= content_weight
    
    style_loss = 0
    for layer_name, gram_source in gram_style.items():
      gm_y = utils.gram_matrix(features_y[layer_name])
      style_loss += loss_fun(gm_y, torch.index_select(gram_source, 0, style_idx))
    style_loss *= style_weight
    
    total_loss = content_loss+style_loss
    
    return {
      "loss" : total_loss,
      "content_loss" : content_loss,
      "style_loss" : style_loss,
    }
  
  return criterion
    
  

def initialize_data_loader(
  folder,
  n_style,
  batch_size,
  num_data_workers,
  big_size=512, images_size=256):
  train_dataset = datasets.ImageFolder(
    folder, training_image_transforms(big_size, images_size)
  )
  
  train_sampler = ElasticDistributedSampler(train_dataset)
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      num_workers=num_data_workers,
      pin_memory=True,
      sampler=train_sampler,
  )
  
  style_idx_dataset = StyleIdxDataset(len(train_dataset), n_style)
  style_idx_sampler = ElasticDistributedSampler(style_idx_dataset)
  style_idx_loader = DataLoader(
    style_idx_dataset,
    batch_size=batch_size,
    num_workers=num_data_workers,
    pin_memory=True,
    sampler=style_idx_sampler,
  )
  
  return train_loader, style_idx_loader

def load_checkpoint(
  checkpoint_file: str,
  device_id: int,
  model: DistributedDataParallel,
  optimizer,
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.
    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """
    state = State(model, optimizer)
    checkpoint_file = Template(checkpoint_file).safe_substitute(EPOCH="last")

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    with tmp_process_group(backend="gloo") as pg:
        rank = dist.get_rank(group=pg)
        
        # get rank that has the largest state.epoch
        epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
        epochs[rank] = state.epoch
        dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
        t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
        max_epoch = t_max_epoch.item()
        max_rank = t_max_rank.item()

        # max_epoch == -1 means no one has checkpointed return base state
        if max_epoch == -1:
            print(f"=> no workers have checkpoints, starting from epoch 0")
            return state

        # broadcast the state from max_rank (which has the most up-to-date state)
        # pickle the snapshot, convert it into a byte-blob tensor
        # then broadcast it, unpickle it and apply the snapshot
        print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

        with io.BytesIO() as f:
            torch.save(state.capture_snapshot(), f)
            raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)

        blob_len = torch.tensor(len(raw_blob))
        dist.broadcast(blob_len, src=max_rank, group=pg)
        print(f"=> checkpoint broadcast size is: {blob_len}")

        if rank != max_rank:
            blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
        else:
            blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

        dist.broadcast(blob, src=max_rank, group=pg)
        print(f"=> done broadcasting checkpoint")

        if rank != max_rank:
            with io.BytesIO(blob.numpy()) as f:
                snapshot = torch.load(f)
            state.apply_snapshot(snapshot, device_id)

        # wait till everyone has loaded the checkpoint
        dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state

def save_checkpoint(state: State, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")

def train(
  train_loader: DataLoader,
  style_idx_loader: DataLoader,
  model: DistributedDataParallel,
  criterion,  # w1 * content_loss + w2 * style_loss
  optimizer,  # SGD,
  epoch: int,
  device_id: int,
  print_freq: int,
):
  
  batch_time = AverageMeter("Time", ":6.3f")
  data_time = AverageMeter("Data", ":6.3f")
  losses = AverageMeter("Loss", ":.4e")
  content_losses = AverageMeter("ContentLoss", ":.2e")
  style_losses = AverageMeter("StyleLoss", ":.2e")
  
  progress = ProgressMeter(
    len(train_loader),
    [batch_time, data_time, losses, content_losses, style_losses],
    prefix="Epoch: [{}]".format(epoch),
  )

  # switch to train mode
  model.train()

  end = time.time()
  for i, ((images, _), style_idx) in enumerate(zip(train_loader, style_idx_loader)):
    # measure data loading time
    data_time.update(time.time() - end)

    images = images.cuda(device_id, non_blocking=True)
    style_idx = style_idx.cuda(device_id, non_blocking=True)
    
    # compute output
    output = model(images, style_idx)
    loss_dict = criterion(output, images, style_idx)
    loss, content_loss, style_loss = [loss_dict[key] for key in ["loss", "content_loss", "style_loss"]]

    # measure accuracy and record loss
    losses.update(loss.item(), images.size(0))
    content_losses.update(content_loss.item(), images.size(0))
    style_losses.update(style_loss.item(), images.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0:
        progress.display(i)

        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)

if __name__ == "__main__":
  main()