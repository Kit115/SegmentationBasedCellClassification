import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models import CellClassifier
from data import ClassificationDataset

def main(args):
    use_cuda    = args.use_cuda and torch.cuda.is_available()
    device      = "cuda:0" if use_cuda else "cpu"

    classifier = CellClassifier(dropout_p=args.dropout_p).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    
    train_dataset    = ClassificationDataset(split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset    = ClassificationDataset(split="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    @torch.no_grad()
    def evaluate():
        classifier.eval()
        
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)
            preds = logits.argmax(-1)
    
            correct += (preds == labels).sum().item()
            total += len(labels)
    
        print(f"Achieved Classification Accuracy of {100 * (correct/total):.2f}% on the test set")
        
        classifier.train()
    
    classifier.train()
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
            
        epoch_loss = 0.
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            logits = classifier(images)
            
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
    
            epoch_loss += loss.item() / len(train_dataloader)

        torch.save({
            "epoch": epoch,
            "model": classifier.state_dict(),
            "optimizer": optimizer.state_dict()
        }, args.checkpoint_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", default="classification_checkpoint.pth")
    
    parser.add_argument("--num-epochs",         type=int,   default=1000)
    parser.add_argument("--batch-size",         type=int,   default=32)
    parser.add_argument("--dropout-p",          type=float, default=0.1)
    parser.add_argument("--learning-rate",      type=float, default=3e-4)

    parser.add_argument('--cuda',    dest="use_cuda", action='store_true')
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)


    args = parser.parse_args()
    main(args)
    
