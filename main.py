import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchinfo import summary

from code.models.multimodal import MultimodalModel
from code.models.compressor import CompressorModel
from code.models.attention import AttentionModel
from code.dataloader import ChDataModule
from code.train import train_epoch, eval_epoch, train_loop


if __name__ == "__main__":
	argparse = argparse.ArgumentParser()
	argparse.add_argument("--model", type=str, default="multimodel")
	argparse.add_argument("--pretrain", type=bool, default=False)
	argparse.add_argument("--use_hrv", type=bool, default=False)
	argparse.add_argument("--use_weather", type=bool, default=False)
	argparse.add_argument("--use_metadata", type=bool, default=False)
	argparse.add_argument("--use_pv", type=bool, default=True)
	argparse.add_argument("--epochs", type=int, default=80)
	argparse.add_argument("--add_epochs", type=int, default=0)
	argparse.add_argument("--batch_size", type=int, default=32)
	argparse.add_argument("--lr", type=float, default=1e-3)
	argparse.add_argument("--weight_decay", type=float, default=0.00)
	argparse.add_argument("--dropout", type=float, default=0.0)
	argparse.add_argument("--batchnorm", type=bool, default=True)
	argparse.add_argument("--checkpoint", type=str, default=None)
	argparse.add_argument("--data_dir", type=str, default="data")
	argparse.add_argument("--dataloader_cfg", type=dict, default={})
	argparse.add_argument("--datamodule_cfg", type=dict, default={})
	argparse.add_argument("--freeze", type=bool, default=False)
	argparse.add_argument("--train", type=bool, default=True)

	args = argparse.parse_args()



	epochs = args.epochs
	datamodule = ChDataModule(args.datamodule_cfg, args.dataloader_cfg)
	datamodule.setup()

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
		train_loop(model, args)

	eval_epoch(args.pretrain)