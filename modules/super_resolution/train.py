import os
import glob
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs import SuperResolutionConfig
from utils.logger import get_logger
from .model import SuperResolutionModel
from .dataset import UKLPDDataset
from .loss import PSRLoss


class Trainer:
    """Super Resolution model trainer."""
    
    def __init__(self, config: SuperResolutionConfig):
        self.config = config
        self.logger = get_logger()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        
        # Initialize model
        self.model = SuperResolutionModel(
            num_blocks=config.num_blocks,
            in_channels=config.in_channels,
            growth_channels=config.growth_channels,
            scale_factor=config.scale_factor
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = PSRLoss(
            alpha=config.alpha,
            beta=1.0 - config.alpha,
            margin=2.0
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Training state
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.best_val_loss = min(self.val_losses) if self.val_losses else float('inf')
            self.logger.info(f"Loaded checkpoint from epoch {self.start_epoch}")
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        num_epochs: int = None
    ):
        """Train the model."""
        num_epochs = num_epochs or self.config.num_epochs
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.start_epoch, num_epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            
            for lr_patches, hr_patches in train_loader:
                lr_patches = lr_patches.to(self.device)
                hr_patches = hr_patches.to(self.device)
                
                self.optimizer.zero_grad()
                sr_patches = self.model(lr_patches)
                loss = self.criterion(sr_patches, hr_patches)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * lr_patches.size(0)
            
            epoch_train_loss = running_loss / len(train_loader.dataset)
            self.train_losses.append(epoch_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for lr_patches, hr_patches in val_loader:
                    lr_patches = lr_patches.to(self.device)
                    hr_patches = hr_patches.to(self.device)
                    sr_patches = self.model(lr_patches)
                    loss = self.criterion(sr_patches, hr_patches)
                    val_loss += loss.item() * lr_patches.size(0)
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            self.val_losses.append(epoch_val_loss)
            self.start_epoch = epoch + 1
            
            # Save checkpoints
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.save_checkpoint(
                    checkpoint_dir / f'best_model_x{self.config.scale_factor}.pth',
                    is_best=True
                )
            
            self.save_checkpoint(
                checkpoint_dir / f'last_model_x{self.config.scale_factor}.pth'
            )
            
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train: {epoch_train_loss:.6f} Val: {epoch_val_loss:.6f}"
            )
    
    def plot_losses(self, save_path: str = None):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    import argparse
    import glob
    import os
    from torch.utils.data import DataLoader
    from configs import SuperResolutionConfig
    from .dataset import UKLPDDataset

    parser = argparse.ArgumentParser(
        description="Train Super Resolution model for license plates"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Directory with HR training images (plates)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory with HR validation images (plates)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/super_resolution",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=800,
        help="Number of epochs",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=8,
        help="Upscaling factor (must match model/config)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="HR patch size for training",
    )
    args = parser.parse_args()

    def list_images(folder):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(folder, e)))
        return sorted(paths)

    train_paths = list_images(args.train_dir)
    val_paths = list_images(args.val_dir)

    if not train_paths:
        raise RuntimeError(f"No training images found in {args.train_dir}")
    if not val_paths:
        raise RuntimeError(f"No validation images found in {args.val_dir}")

    train_dataset = UKLPDDataset(
        train_paths,
        patch_size=args.patch_size,
        scale_factor=args.scale_factor,
    )
    val_dataset = UKLPDDataset(
        val_paths,
        patch_size=args.patch_size,
        scale_factor=args.scale_factor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    sr_config = SuperResolutionConfig(
        device=args.device,
        num_blocks=6,
        in_channels=3,
        growth_channels=64,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = Trainer(sr_config)

    # resume
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)

    trainer.train(train_loader, val_loader, num_epochs=args.epochs)