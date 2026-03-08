import io
from dataclasses import dataclass, field
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import wandb
from models import plain_20, plain_44, resnet_20, resnet_56, resnet_44

# --------------------------------------------------------
# 1. Config
# --------------------------------------------------------
@dataclass
class CFG:
    data_parquet: str = "data/data.parquet"   # merged parquet containing both train/test with labels
    train_meta_csv: str = "data/train.csv"    # metadata only (still useful for IDs / splits)
    test_meta_csv: str = "data/test.csv"

    # Column names
    id_col: str = "id"
    image_col: str = "image"
    label_col: str = "label"

    # Fashion-MNIST shape
    H: int = 28
    W: int = 28
    num_classes: int = 10

    # Training
    seed: int = 42
    train_ratio: float = 0.9
    max_train_samples: int = None  
    max_eval_samples: int = None

    # Optim / trainer 
    optimizer: str = "SGD"
    lr: float = 0.1  
    weight_decay: float = 0.0001             
    momentum: float = 0.9         
    epochs: int = 20              
    batch_size: int = 128 
    lr_milestones: Tuple[int, ...] = field(default_factory=lambda: (10, 15)) 
    lr_gamma: float = 0.1        
    
    # Wandb 
    logging_steps: int = 50
    wandb_project: str = "DL-HA1-FashionMNIST-ResNet"

cfg = CFG()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(cfg.seed)

# --------------------------------------------------------
# 2. Dataset
# --------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomCrop(cfg.H, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode_image(img_any, target_H, target_W):
    if isinstance(img_any, PILImage.Image):
        img = img_any
    elif isinstance(img_any, dict):

        if img_any.get("bytes", None) is not None:
            img = PILImage.open(io.BytesIO(img_any["bytes"]))
        elif img_any.get("path", None) is not None:
            img = PILImage.open(img_any["path"])
        else:
            raise ValueError(f"unknown image format: {img_any.keys()}")
    else:
        img = PILImage.fromarray(np.array(img_any, dtype=np.uint8))


    return img.convert("L").resize((target_W, target_H))

class FashionMNIST_HF_Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        img = decode_image(item[cfg.image_col], cfg.H, cfg.W)
        label = int(item[cfg.label_col])

        if self.transform:
            img = self.transform(img)

        return img, label

# --------------------------------------------------------
# 3. train & evaluate function
# --------------------------------------------------------
def train_and_evaluate(model, model_name, train_loader, val_loader, cfg, device):

    wandb.init(
        project=cfg.wandb_project,
        name=f"{model_name}-run",
        config={
            "model": model_name,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "momentum": cfg.momentum,
            "dataset": "FashionMNIST-Resplit"
        }
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total_params})
    print(f"[{model_name}] training begins, params: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)

    best_val_acc = 0.0
    
    for epoch in range(cfg.epochs):

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({'Loss': f"{train_loss/train_total:.4f}"})

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]")
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']


        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "Train Loss": epoch_train_loss,
            "Train Acc": epoch_train_acc,
            "Val Loss": epoch_val_loss,
            "Val Acc": epoch_val_acc
        })
        

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    print(f"\n[{model_name}] training finished! Best validation accuracy: {best_val_acc:.2f}%")
    
    
    return model

# --------------------------------------------------------
# 4. main function to run all models
# --------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds_all = load_dataset("parquet", data_files=cfg.data_parquet)["train"]
    train_meta = pd.read_csv(cfg.train_meta_csv)
    test_meta = pd.read_csv(cfg.test_meta_csv)

    train_ids = set(train_meta[cfg.id_col])
    test_ids = set(test_meta[cfg.id_col])
    
    ds_train_original = ds_all.filter(lambda x: x[cfg.id_col] in train_ids)
    ds_test  = ds_all.filter(lambda x: x[cfg.id_col] in test_ids)
    
    split_result = ds_train_original.train_test_split(test_size=0.1, seed=cfg.seed)
    ds_train = split_result['train']
    ds_val = split_result['test']

    print(f"training set size: {len(ds_train)} | test set size: {len(ds_test)}")
        
    train_loader = DataLoader(FashionMNIST_HF_Dataset(ds_train, transform=train_transform), 
                              batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(FashionMNIST_HF_Dataset(ds_val, transform=eval_transform), 
                            batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(FashionMNIST_HF_Dataset(ds_test, transform=eval_transform), 
                             batch_size=cfg.batch_size, shuffle=False, num_workers=2)        
    models_to_train = {
        "Plain-20": plain_20(num_classes=cfg.num_classes, input_channels=1),
        "Plain-44": plain_44(num_classes=cfg.num_classes, input_channels=1),
        "ResNet-20": resnet_20(num_classes=cfg.num_classes, input_channels=1),
        "ResNet-44": resnet_44(num_classes=cfg.num_classes, input_channels=1),
        "ResNet-56": resnet_56(num_classes=cfg.num_classes, input_channels=1)
    }
    

    for name, model in models_to_train.items():
        print(f"\n{'='*50}\ntraining begins: {name}\n{'='*50}")
        
        trained_model = train_and_evaluate(model, name, train_loader, val_loader, cfg, device)
        
        print(f"testing {name} ...")

        best_model = trained_model 
        best_model.load_state_dict(torch.load(f"best_{name}.pth", weights_only=True))
        best_model.eval()
        
        test_loss = 0.0     
        test_correct = 0
        test_total = 0
        criterion = nn.CrossEntropyLoss() 

        test_targets = []
        test_preds = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = best_model(inputs)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                test_targets.extend(targets.cpu().numpy())
                test_preds.extend(predicted.cpu().numpy())
                
        final_test_loss = test_loss / test_total
        final_test_acc = 100. * test_correct / test_total
        print(f">>> {name} Test Loss: {final_test_loss:.4f} | acc: {final_test_acc:.2f}%\n")
        
        wandb.run.summary["Final_Test_Loss"] = final_test_loss
        wandb.run.summary["Final_Test_Accuracy"] = final_test_acc
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        wandb.log({"Test_Confusion_Matrix": wandb.plot.confusion_matrix(
            y_true=test_targets, 
            preds=test_preds, 
            class_names=class_names
        )})
        wandb.finish()

if __name__ == "__main__":
    main()