import torch
import numpy as np

from models import NucleusSegmenter

from data import SegmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):

    use_cuda = args.use_cuda and torch.cuda.is_available() 
    device = "cuda:0" if use_cuda else "cpu"

    classifier = NucleusSegmenter().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if args.resume:
        checkpoint = torch.load(args.checkpoint_name, weights_only=False, map_location="device")
        start_epoch = checkpoint["epoch"] + 1
        classifier.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 1
    
    dataloader_workers = multiprocessing.cpu_count() // 2 if use_cuda else 0
    dataset = SegmentationDataset(split="train", device=device, num_positives=args.num_positives, num_negatives=args.num_negatives)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=use_cuda, num_workers=dataloader_workers)

    epoch_losses = []
    window_size  = 250

    tqdm_iterator = tqdm(range(start_epoch, args.num_epochs + 1), desc=f"Epoch: #{0:05d}; Loss: {0.:.8f}")

    for epoch in tqdm_iterator:
        epoch_loss = 0.0
        for images, queries, labels in dataloader:
            images, queries, labels = images.to(device, non_blocking=use_cuda), queries.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda)
            logits = classifier(queries, images).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() / len(dataloader)

        epoch_losses.append(epoch_loss)

        moving_average = sum(epoch_losses[-window_size:]) / min(window_size, len(epoch_losses))
        tqdm_iterator.set_description(f"Epoch: #{epoch:05d}; Loss: {moving_average:.8f}")

        torch.save({
            "epoch": epoch,
            "model": classifier.state_dict(),
            "optimizer": optimizer.state_dict()
        }, args.checkpoint_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", default="checkpoints/segmentation_checkpoint.pth")
    parser.add_argument("--num-positives", default=128)
    parser.add_argument("--num-negatives", default=256)
    
    parser.add_argument("--num-epochs", default=2000)
    parser.add_argument("--batch-size", default=12)
    
    parser.add_argument('--resume',    dest="resume", action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=False)

    parser.add_argument('--cuda',    dest="use_cuda", action='store_true')
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)


    args = parser.parse_args()
    main(args)

