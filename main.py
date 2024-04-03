import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchinfo import summary

from code.models.multimodal import MultimodalModel
from code.models.compressor import CompressorModel
from code.models.attention import AttentionModel
from code.dataloader import ChDataModule, ChCacheDataset
from code.train import train_epoch, eval_epoch, train_loop


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--model", type=str, default="multimodal")
	argparser.add_argument("--pretrain", type=bool, default=False)
	argparser.add_argument("--use_hrv", type=bool, default=True)
	argparser.add_argument("--use_weather", type=bool, default=False)
	argparser.add_argument("--use_metadata", type=bool, default=False)
	argparser.add_argument("--use_pv", type=bool, default=True)
	argparser.add_argument("--epochs", type=int, default=80)
	argparser.add_argument("--add_epochs", type=int, default=0)
	argparser.add_argument("--batch_size", type=int, default=8)
	argparser.add_argument("--lr", type=float, default=1e-3)
	argparser.add_argument("--weight_decay", type=float, default=0.00)
	argparser.add_argument("--dropout", type=float, default=0.0)
	argparser.add_argument("--batchnorm", type=bool, default=True)
	argparser.add_argument("--checkpoint", type=str, default=None)
	argparser.add_argument("--data_dir", type=str, default="data")
	argparser.add_argument("--dataloader_cfg", type=dict, 
		default={"num_workers": 8, "batch_size": 8, "pin_memory": True, "persistent_workers": True})
	argparser.add_argument("--datamodule_cfg", type=dict, 
		default={"val_split": 0.1, "cache_dir": "data/cache/"})
	argparser.add_argument("--cached_data", type=bool, default=True)
	argparser.add_argument("--freeze", type=bool, default=False)
	argparser.add_argument("--train", type=bool, default=True)
	args = argparser.parse_args()


	datamodule = ChDataModule(args.datamodule_cfg, args.dataloader_cfg)
	datamodule.setup('fit')
	
	if args.cached_data:
		train_loader = datamodule.train_dataloader_cached()
		val_loader = datamodule.val_dataloader_cached()
	else:
		train_loader = datamodule.train_dataloader()
		val_loader = datamodule.val_dataloader()

	criterion = nn.L1Loss()

	if args.model == "multimodal":
		model = MultimodalModel(args)
		summary(model)

	elif args.model == "compressor":
		model = CompressorModel(args)
		summary(model)

	elif args.model == "attention":
		model = AttentionModel(args)
		summary(model)

	else:
		raise ValueError("Invalid model")
	
	if args.checkpoint is not None:
		model.load_state_dict(torch.load(args.checkpoint))
	
	if args.freeze:
		model.freeze_pretrain()

	if args.train:
		train_loop(model, args, train_loader, val_loader)

	eval_epoch(model, args, criterion, val_loader)