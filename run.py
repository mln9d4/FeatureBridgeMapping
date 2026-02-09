# import all necessary libraries

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer 
from mmdet.models import HEADS
from mmdet3d.unibev_plugin.models.dense_heads import bev_consumer
import matplotlib.pyplot as plt
from models.network import Network
from config.config import config
from data.dataset import BEVFeaturesDataset, PaddDataset
from animation import create_animation
print("Libraries imported successfully")
# print(torch.__version__)

# Select Cuda Device to work on
torch.manual_seed(42)

device_number = "0"
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def bvc_to_bchw(tensor, H=200, W=200):
    """
    Convert [B, V, C] -> [B, C, H, W].
    Accepts torch.Tensor or numpy.ndarray. If H and W don't match V, will try to infer square side.
    """
    B, V, C = tensor.shape
    if V != H*W:
        raise ValueError(f"H and W are not product from V, please specify what H and W you want")
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.reshape(B, H, W, C)
        tensor = tensor.permute(0, 3, 2, 1).contiguous()
        return tensor
    elif isinstance(tensor, np.ndarray):
        tensor = tensor.copy()
        tensor = tensor.reshape(B, H, W, C)
        tensor = np.transpose(tensor, (0, 3, 2, 1))
        return tensor
    else:
        raise TypeError("Expected torch.TEnsor or numpy.ndarray")

def save_loss_gradient_map(preds, targets, model, epoch, batch_idx):
    """
    Visualizes where the loss is pulling the model.
    Bright spots = high influence.
    """
    # 1. We need the gradient of the loss with respect to the prediction
    # We must ensure the prediction has grad enabled
    preds_for_grad = preds.detach().requires_grad_(True)
    
    # 2. Re-calculate loss for this specific pair
    loss_dict = model.loss(preds_for_grad, targets)
    loss = loss_dict['bev_consumer_loss_std_weighted_l1loss']
    
    # 3. Calculate gradients: dLoss / dPreds
    grad = torch.autograd.grad(loss, preds_for_grad)[0]
    
    # 4. Reshape to spatial (B, C, H, W) and take mean over channels
    bs, n, c = grad.shape
    grad_spatial = grad.transpose(1, 2).view(bs, c, 200, 200) # Using your bev_h/w
    grad_map = grad_spatial[0].cpu().numpy() # First sample in batch

    # 5. Plot with Robust Scaling (to avoid the "one color" issue)
    plt.figure(figsize=(6, 6))
    v_min, v_max = np.percentile(grad_map[0], [2, 98]) # Clip outliers
    
    plt.imshow(grad_map[0], cmap='magma', vmin=v_min, vmax=v_max)
    plt.colorbar(label="Gradient Intensity")
    plt.title(f"Loss Gradient Map (Spatial Influence)\nEpoch {epoch}")
    
    # Create the directory if it doesn't exist
    save_dir = "/home/mingdayang/palette_diffusion/figures"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(save_dir, f"spatial_influence_ep{epoch}_b{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()

def save_sample_output(preds, targets, epoch, batch_idx):
    """
    Visualizes the model's output vs the target. Visualize channel 0
    """
    # Assuming preds and targets are in [B, C, H, W] format
    pred_map = preds[0].cpu().numpy() # First sample in batch
    target_map = targets[0].cpu().numpy() # First sample in batch

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pred_map[0], cmap='viridis') # First channle

    plt.colorbar()
    plt.title(f"Model Output\nEpoch {epoch}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(target_map[0], cmap='viridis') 

    plt.colorbar()
    plt.title(f"Target\nEpoch {epoch}")
    
    # Create the directory if it doesn't exist
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(save_dir, f"model_output_vs_target_ep{epoch}_b{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()

def load_data():
    # Load the saved data
    dataset = BEVFeaturesDataset(root_dir='/home/mingdayang/palette_diffusion/data/bev_features', transform=PaddDataset(pad_size=8))

    
    return dataset

def create_splits(dataset, train_split=0.8):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_split * len(dataset)), len(dataset) - int(train_split * len(dataset))])

    return train_dataset, test_dataset

def make_loader(batch_size, dataset):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Let's check out what we've created

    return dataloader

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="manual-training-notebook", config=hyperparameters) as run:
        # access all HPs through run.config, so logging matches execution.
        config = run.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, optimizer, scheduler = make(config)

        # and use them to train the model
        train(model, train_loader, test_loader, optimizer, scheduler, config)
        save_model(model, config)

        return model

def make(config):
    # 1. Data Loading
    dataset = load_data()
    # train_dataset, test_dataset = create_splits(dataset, train_split=0.8)
    sub_dataset = torch.utils.data.Subset(dataset, list(range(0,4)))
    # train_loader = make_loader(config.batch_size, train_dataset)
    # test_loader = make_loader(config.batch_size, test_dataset)
    train_loader = make_loader(config.batch_size, sub_dataset)
    test_loader = make_loader(config.batch_size, sub_dataset)

    # 2. Model Setup
    model = build_model(config)

    # 3. Optimizer (Standard AdamW with weight decay)
    # Palette often uses a lower LR like 5e-5 for 256x256 images
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )

    # 4. Scheduler (Warmup then Constant)
    # We use LinearLR for the warmup and then 'switch' to constant
    warmup_steps = config.warmup_steps # Typically 5000-10000 iterations
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=config.start_factor, # Start at 0.1% of max LR
        end_factor=config.end_factor, 
        total_iters=warmup_steps
    )
    
    return model, train_loader, test_loader, optimizer, scheduler

def build_model(config):
    model = Network(config.model, config.beta_schedule)
    model.set_new_noise_schedule(device=device, phase='train')
    model.set_loss(nn.L1Loss())
    return model

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
    

def train(model, train_loader, test_loader, optimizer, scheduler, config):
    run = wandb.init(project="manual-training-notebook", config=config)
    run.watch(model, log="all", log_freq=10)
    train_time_start_on_cpu = timer()
    model.to(device)
    for epoch in tqdm(range(config.epochs)):
        test_loss = 0.0
        train_loss =0.0

        model.train()

        print(f"Epoch {epoch}/{config.epochs}\n-------------------------------")

        for batch_idx, sample in enumerate(train_loader):
            X, y = sample['img_bev_embed'], sample['pts_bev_embed']
            # Move data to device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            # print(X.shape, y.shape)
            loss = model(y_0=y, y_cond=X)

            # 2. Compute loss
            # not needed in diffusion
            
            train_loss += loss.item()
            # 3. Zero the gradients
            optimizer.zero_grad()

            # 4. Backward pass
            loss.backward() 

            # if epoch % 10 == 0 and batch_idx % 20 == 0:
            #     save_loss_gradient_map(outputs, y, model, epoch=epoch, batch_idx=batch_idx)
                # save_activation_map(model, X, epoch=epoch)

            # 5. Optimizer step
            optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            run.log({"learning_rate": current_lr})

            if batch_idx % 3 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)


        ### validation los
        if epoch % config.val_interval == 0:
            model.eval()
            with torch.inference_mode():
                total = len(test_loader)
                for i, sample in enumerate(test_loader):
                    X, y = sample['img_bev_embed'], sample['pts_bev_embed']
                    X, y = X.to(device), y.to(device)
                    outputs, ret_arr = model.ddim_restoration(y_cond=X, sample_num=20, step=100, eta=0.0)
                    # print(outputs.shape)
                    # loss = criterion(outputs, y)
                    loss = model.loss_fn(outputs, y)
                    print(f"Validation Loss {i}/{total}: {loss.item():.4f}")
                    test_loss += loss.item()
                    create_animation(ret_arr, batch=batch_idx, epoch=epoch, val_i=i)


                save_sample_output(outputs, y, epoch=epoch, batch_idx=batch_idx)
                avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")
        run.log({"train_loss": avg_train_loss, "test_loss": avg_test_loss, "epoch": epoch})

        if epoch % config.save_interval == 0:
            save_model(model, config, epoch)

    run.finish()
    train_time_end_on_cpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model.parameters()).device))
    return model

def save_model(model, config, epoch=None):
    MODEL_PATH = os.path.join(os.getcwd(), 'models', config.architecture)
    os.makedirs(MODEL_PATH, exist_ok=True)
    if epoch is None:
        MODEL_SAVE_PATH = os.path.join(MODEL_PATH, f"{config.architecture}_model_last_epoch.pth")
    else:
        MODEL_SAVE_PATH = os.path.join(MODEL_PATH, f"{config.architecture}_model_epoch{epoch}.pth")

    temp_ = {}
    temp_['model_type'] = config.architecture
    temp_['model_config'] = config.model
    temp_['state_dict'] = model.state_dict()

    torch.save(temp_, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
def main():
    model = model_pipeline(config)




if __name__=="__main__":
    main()