import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import cv2
import os

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from data_utils import ColmapDataset

@dataclass
class TrainConfig:
    num_epochs: int = 200
    batch_size: int = 1
    learning_rate: float = 0.01
    grad_clip: float = 1.0
    save_every: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    debug_every: int = 1  # Save debug images every N epochs
    debug_samples: int = 1  # Number of images to save for debugging

class GaussianTrainer:
    def __init__(
        self, 
        model: GaussianModel,
        renderer: GaussianRenderer,
        config: TrainConfig,
        device: torch.device
    ):
        self.model = model.to(device)
        self.renderer = renderer.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer
        optable_params = [
            {'params': [self.model.positions], 'lr': 0.000016, "name": "xyz"},
            {'params': [self.model.colors], 'lr': 0.025, "name": "color"},
            {'params': [self.model.opacities], 'lr': 0.05, "name": "opacity"},
            {'params': [self.model.scales], 'lr': 0.005, "name": "scaling"},
            {'params': [self.model.rotations], 'lr': 0.001, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(optable_params, lr=0.001, eps=1e-15)
        
        # Create checkpoint and log directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)
        
        # Keep track of debug indices
        self.debug_indices = None

    def save_debug_images(self, epoch: int, rendered_images: torch.Tensor, 
                         gt_images: torch.Tensor, image_paths: list):
        """
        Save debug images comparing ground truth and rendered results
        """
        # Convert tensors to numpy arrays
        rendered = rendered_images.detach().cpu().numpy()
        gt = gt_images.detach().cpu().numpy()
        epoch_dir = Path(self.config.log_dir) / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=True)
        
        for b in range(rendered.shape[0]):
            base_name = Path(image_paths[b]).stem
            rendered_img = (rendered[b] * 255).clip(0, 255).astype(np.uint8)
            gt_img = (gt[b] * 255).clip(0, 255).astype(np.uint8)
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            comparison = np.concatenate([gt_img, rendered_img], axis=1)
            output_path = epoch_dir / f"{base_name}.png"
            cv2.imwrite(str(output_path), comparison)

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{epoch:06d}.pt"
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def visualize_rendering(self, dataset, save_vid_path: str, num_frames: int = 300):
        """
        Create a video of the scene rendered from the original camera path
        
        Args:
            dataset: ColmapDataset instance containing camera parameters
            save_vid_path: Path to save the output video
            num_frames: Number of frames in the circular path
        """
        print("Generating rendering visualization...")
        
        # Get sample K matrix and image dimensions
        sample = dataset[0]
        K = sample['K'].to(self.device)
        H, W = sample['image'].shape[:2]
        
        # Initialize video writer
        out = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (W*2, H))
        
        # Get gaussian parameters (only need to compute once)
        with torch.no_grad():
            gaussian_params = self.model()
        
        # Render frames
        for data_item in tqdm(dataset, desc="Rendering frames"):
            # Convert camera poses to torch tensors
            R_torch = data_item['R'].to(self.device)
            t_torch = data_item['t'].to(self.device).reshape(-1, 3)
            
            # Render frame
            with torch.no_grad():
                rendered_image = self.renderer(
                    means3D=gaussian_params['positions'],
                    covs3d=gaussian_params['covariance'],
                    colors=gaussian_params['colors'],
                    opacities=gaussian_params['opacities'],
                    K=K.squeeze(0),
                    R=R_torch.squeeze(0),
                    t=t_torch.squeeze(0),
                )
            
            # Convert to numpy and BGR format for OpenCV
            frame = rendered_image.cpu().numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            ori_img = (data_item['image']*255).cpu().numpy().astype(np.uint8)
            vis = cv2.cvtColor(np.concatenate((ori_img, frame), axis=1), cv2.COLOR_RGB2BGR)
            # Write frame
            out.write(vis)
        
        # Release video writer
        out.release()
        print(f"Video saved to: {save_vid_path}")
        

    def train_step(self, batch: dict, in_train = True) -> float:
        """Single training step"""
        # Get batch data and prepare camera matrices
        images = batch['image'].to(self.device)            # (B, H, W, 3)
        K = batch['K'].to(self.device)                     # (B, 3, 3)
        R = batch['R'].to(self.device)                     # (B, 3, 3)
        t = batch['t'].to(self.device).reshape(-1, 3)      # (B, 3)
        
        # Forward pass
        gaussian_params = self.model()
        rendered_images = self.renderer(
            means3D=gaussian_params['positions'],
            covs3d=gaussian_params['covariance'],
            colors=gaussian_params['colors'],
            opacities=gaussian_params['opacities'],
            K = K.squeeze(0),
            R = R.squeeze(0),
            t = t.squeeze(0),
        )
        rendered_images = rendered_images.unsqueeze(0)
        
        if not in_train:
            return rendered_images

        # Compute RGB loss
        loss = torch.abs(rendered_images - images).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.grad_clip
        )
        
        # Optimization step
        self.optimizer.step()
        
        return loss.item(), rendered_images

    def train(self, train_loader: DataLoader):
        """Main training loop"""
        # Select fixed indices for debugging
        if self.debug_indices is None:
            dataset_size = len(train_loader.dataset)
            self.debug_indices = np.random.choice(
                dataset_size, 
                min(self.config.debug_samples, dataset_size), 
                replace=False
            )
        
        for epoch in range(self.config.num_epochs):
            # Training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss, rendered_images = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Update progress bar
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch)
                
            # Save debug images every N epochs
            if epoch % self.config.debug_every == 0:
                debug_batches = []
                for idx in self.debug_indices:
                    debug_batches.append(train_loader.dataset[idx])
                
                # Stack debug batches
                debug_batch = {
                    k: torch.stack([b[k] for b in debug_batches], 0) 
                    if torch.is_tensor(debug_batches[0][k])
                    else [b[k] for b in debug_batches]
                    for k in debug_batches[0].keys()
                }
                
                # Get rendered images for debug batch
                with torch.no_grad():
                    debug_rendered = self.train_step(debug_batch, in_train=False)
                
                # Save debug images
                self.save_debug_images(
                    epoch=epoch,
                    rendered_images=debug_rendered,
                    gt_images=debug_batch['image'],
                    image_paths=debug_batch['image_path']
                )

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting')
    
    # Data paths
    parser.add_argument('--colmap_dir', type=str, required=True,
                      help='Directory containing COLMAP data (with sparse/0/ and images/)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    # Debug parameters
    parser.add_argument('--debug_every', type=int, default=1,
                      help='Save debug images every N epochs')
    parser.add_argument('--debug_samples', type=int, default=1,
                      help='Number of images to save for debugging')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create config
    config = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=os.path.join(args.checkpoint_dir, "debug_images"),
        debug_every=args.debug_every,
        debug_samples=args.debug_samples
    )
    
    # Initialize dataset
    dataset = ColmapDataset(args.colmap_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get image dimensions from dataset
    sample = dataset[0]['image']
    H, W = sample.shape[:2]
    
    # Initialize model using COLMAP points
    model = GaussianModel(
        points3D_xyz=dataset.points3D_xyz,
        points3D_rgb=dataset.points3D_rgb
    )
    
    # Initialize renderer
    renderer = GaussianRenderer(
        image_height=H,
        image_width=W
    )
    
    # Initialize trainer
    trainer = GaussianTrainer(model, renderer, config, device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        config.num_epochs -= start_epoch
    
    
    # Start training
    print("Starting training...")
    print(f"Training on {len(dataset)} images for {config.num_epochs} epochs")
    print(f"Debug images will be saved every {config.debug_every} epochs")
    print(f"Using {config.debug_samples} debug samples")
    trainer.train(train_loader)
    print("Training completed!")

    trainer.visualize_rendering(dataset, os.path.join(args.checkpoint_dir, "debug_rendering.mp4"))

if __name__ == "__main__":
    main()
