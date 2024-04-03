import time
import torch
from torch import nn
from torch import optim
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, args, optimiser, criterion, train_loader):
    model.train()
    running_loss = 0.0
    count = 0
    for i, d in enumerate(train_loader):

        optimiser.zero_grad()

        if args.cached_data:
            (pv_features, metadata_features, hrv_data, pv_targets) = d
            weather_data = torch.Tensor([])
        elif args.pretrain and not args.use_hrv:
            (pv_features, latitude, longitude, day_of_year, time_of_day,
              orientation, tilt, kwp, pv_targets) = d
            hrv_data = torch.Tensor([])
            weather_data = torch.Tensor([])
            metadata_features = torch.stack((latitude, longitude, day_of_year, time_of_day, orientation, tilt, kwp), axis=1)
        else:
            (pv_features, latitude, longitude, day_of_year, time_of_day,
              orientation, tilt, kwp, hrv_data, pv_targets) = d
            weather_data = torch.Tensor([])
            metadata_features = torch.stack((latitude, longitude, day_of_year, time_of_day, orientation, tilt, kwp), axis=1)

        predictions = model(
            pv_features.to(device, dtype=torch.float),
            hrv_data.to(device, dtype=torch.float),
            weather_data.to(device, dtype=torch.float),
            metadata_features.to(device, dtype=torch.float)
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if float(loss) == float('nan') or running_loss == float('nan'):
            print(predictions)
            raise FloatingPointError

        loss.backward()
        optimiser.step()
        if (i + 1) % (200) == 0:
            print(f"{i+1}: {running_loss / count}")

    print(f"Train Loss: {running_loss / count}")
    return running_loss / count

def eval_epoch(model, args, criterion, val_loader):
    model.eval()
    running_loss = 0.0
    count = 0
    for i, d in enumerate(val_loader):

        if args.cached_data:
            (pv_features, metadata_features, hrv_data, pv_targets) = d
            weather_data = torch.Tensor([])
        elif args.pretrain and not args.use_hrv:
            (pv_features, latitude, longitude, day_of_year, time_of_day,
              orientation, tilt, kwp, pv_targets) = d
            hrv_data = torch.Tensor([])
            weather_data = torch.Tensor([])
            metadata_features = torch.stack((latitude, longitude, day_of_year, time_of_day, orientation, tilt, kwp), axis=1)
        else:
            (pv_features, latitude, longitude, day_of_year, time_of_day,
              orientation, tilt, kwp, hrv_data, pv_targets) = d
            weather_data = torch.Tensor([])
            metadata_features = torch.stack((latitude, longitude, day_of_year, time_of_day, orientation, tilt, kwp), axis=1)

        predictions = model(
            pv_features.to(device, dtype=torch.float),
            hrv_data.to(device, dtype=torch.float),
            weather_data.to(device, dtype=torch.float),
            metadata_features.to(device, dtype=torch.float)
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))


        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size
    print(f"Val Loss: {running_loss / count}")
    return running_loss / count

def train_loop(model, args, train_loader, val_loader):
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    os.makedirs('checkpoints', exist_ok=True)

    best_val = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        print("Epoch", epoch + 1)
        train_loss = train_epoch(model, args, optimiser, criterion, train_loader)
        val_loss = eval_epoch(model, args, criterion, val_loader)

        t1 = time.time()
        print("Train Loss:", train_loss, "Val Loss:", val_loss, "LR:", optimiser.param_groups[0]['lr'], "Time:", t1-t0)

        torch.save(model.state_dict(), f"checkpoints/checkpoint_{args.name}_epoch{epoch}.pt")
        if val_loss < best_val:
            print("Saving new best")
            best_val = val_loss
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{args.name}_best.pt")
        print("")