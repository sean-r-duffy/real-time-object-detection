"""
Nick Cantalupa and Sean Duffy
Computer Vision Spring 2025

This file generates an adversarial patch for the ResNet model when run as a script.
It uses the example images in the data directory to create an adversarial patch.
The adversarial patch is saved as a pickle file and a PNG image.

Must run with --target <target_class_index> to specify the target class index.
The target class index is the index of the class in the ImageNet dataset.
"""

import argparse
import os
import pickle
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
torch.random.manual_seed(0)


class ImageDataset(Dataset):
    """
    Class for loading example images and applying transformation, given to a dataloader

    :ivar image_dir: Path to the directory containing image files.=
    :ivar image_files: List of image file names with `.jpeg` or `.jpg` extensions
        found within the specified directory.
    :ivar transform: Optional transformations to be applied to each image, consisting
        of resizing, tensor conversion, and normalization if not otherwise specified.
    """

    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None) -> None:
        """
        Initializes a dataset object for loading and processing image files from a given directory.
        The dataset loads image files with extensions `.jpeg` or `.jpg` from the specified directory
        and applies transformations for resizing, converting to tensor, and normalizing, if not specified.

        :param image_dir: Path to the directory containing the image files to be loaded.
        :param transform: Optional set of transformations to apply to each image. Defaults to a
            set of transformations including resizing, conversion to tensor, and normalization.=
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        :param idx: Index of the image in the dataset directory.
        :return: Transformed image tensor corresponding to the specified index.
        """
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_image_dataloader(batch_size: int = 4, shuffle: bool = True, num_workers: int = 6,
                         image_dir: str = 'data/example_images') -> DataLoader:
    """
    Returns a dataloader for the ImageDataset class

    :param batch_size: Size of each data batch.
    :param shuffle: Whether the data should be shuffled after each epoch.
    :param num_workers: Number of subprocesses to use for data loading.
    :param image_dir: Path to the directory containing image data.
    :return: A PyTorch DataLoader for processing the image dataset.
    """
    dataset = ImageDataset(image_dir=image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


class PatchApplier(nn.Module):
    """
    Applies a patch to a batch of images.

    :ivar image_size: The (height, width) dimensions of input images.
    :type image_size: tuple[int, int]
    """

    def __init__(self, image_size):
        super(PatchApplier, self).__init__()
        self.image_size = image_size

    def forward(self, images, patch):
        """
        Given a patch tensor and an images tensor, apply the patch to the image and return the patched image tensor.
        Applied at a random location within the image, with random rotation and scaling

        :param images: Input tensor of batched images (batch_size, channels, image_height, image_width)
        :param patch: Patch tensor to be transformed and applied to the images (channels, patch_height, patch_width).
        :return: A tensor of images with the transformed patch applied (batch_size, channels, image_height, image_width)
        """
        batch_size = images.size(0)
        patch_h, patch_w = patch.shape[1:3]
        img_h, img_w = self.image_size

        patched_images = images.clone()

        # Get random rotation and scale for each image
        angles = torch.randint(-30, 30, (batch_size,), device=device)
        scales = 0.8 + 0.4 * torch.rand(batch_size, device=device)

        # Iterate through batch and apply patch to each image
        for i in range(batch_size):
            angle = angles[i].item()
            scale = scales[i].item()

            # Apply rotation and scaling
            rotated_patch = transforms.functional.rotate(patch, angle)
            scaled_size = (int(patch_h * scale), int(patch_w * scale))
            scaled_patch = transforms.functional.resize(rotated_patch, scaled_size)

            # Overlay the patch on the image
            max_x = img_w - scaled_size[1]
            max_y = img_h - scaled_size[0]
            if max_x > 0 and max_y > 0:
                x_location = torch.randint(0, max_x, (1,), device=device).item()
                y_location = torch.randint(0, max_y, (1,), device=device).item()
                patched_images[i, :, y_location:y_location + scaled_size[0],
                x_location:x_location + scaled_size[1]] = scaled_patch

        return patched_images


def visualize_patch(patch_tensor: torch.Tensor) -> None:
    """
    Helper function to visualize a patch tensor, for use in debugging

    :param patch_tensor: The input tensor representing the adversarial patch (C, H, W)
    :return: None
    """
    # Clone the tensor and detach from GPU
    patch_np = patch_tensor.clone().detach().cpu().numpy()

    # Transpose from [C, H, W] to [H, W, C] for displaying
    patch_np = np.transpose(patch_np, (1, 2, 0))

    patch_np = np.clip(patch_np, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(patch_np)
    plt.axis('off')
    plt.title('Adversarial Patch')
    plt.show()


def create_adversarial_patch(model: nn.Module, target_class: int, image_dir: str,
                             image_size: tuple[int, int] = (224, 224),
                             patch_size: tuple[int, int] = (64, 64), learning_rate: float = 0.1,
                             num_iterations: int = 100, batch_size: int = 10) -> torch.Tensor:
    """
    Generates an adversarial patch targeting a specific class for a given model

    :param model: Model to be used for generating the adversarial patch
    :param target_class: Target class index for the patch
    :param image_dir: Directory containing training images to which the patch will be applied
    :param image_size: Dimensions of the input image: (height, width)
                       Defaults to (224, 224)
    :param patch_size: Dimensions of the patch: (height, width)
                       Defaults to (64, 64)
    :param learning_rate: Learning rate for the optimization process. Defaults to 0.1
    :param num_iterations: Training epochs. Defaults to 100
    :param batch_size: Batch size for loading training images. Defaults to 10

    :return: A tensor representing the optimized adversarial patch
    """
    patch = torch.rand(3, patch_size[0], patch_size[1], requires_grad=True,
                       device=device)  # Initialize patch with uniform random noise
    optimizer = optim.Adam([patch], lr=learning_rate)  # Only updating the patch
    loss_func = nn.CrossEntropyLoss()

    # Set target tensor
    target = torch.zeros((batch_size, 1000), device=device)
    target[:, target_class] = 1

    # Setup data loading and patch application
    dataloader = get_image_dataloader(batch_size=batch_size, image_dir=image_dir, num_workers=2)
    patch_applier = PatchApplier(image_size)

    model.eval()
    for iteration in tqdm(range(num_iterations)):
        for images in dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            images = patch_applier(images, patch)
            outputs = model(images)

            loss = loss_func(outputs, target)  # Negative because we want to maximize error

            # Backward pass
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.data.clamp_(0, 1)  # Keep patch values within [0, 1] range

        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}")
        visualize_patch(patch.detach())

    return patch.detach()


if __name__ == "__main__":
    # Command line argument parsing for target class index
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, help="Target class index", required=True)
    args = parser.parse_args()

    print(f"Using device: {device}")  # Show device being used (cuda or cpu)
    # Load model
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    class_names = weights.meta["categories"]  # Use for debugging if necessary

    example_img_dir = 'data/example_images'  # Training image directory
    target_class = args.target

    # Calculate patch
    patch = create_adversarial_patch(model, target_class, example_img_dir, patch_size=(128, 128))

    # Save patch
    with open("adversarial_patch.pkl", "wb") as f:
        pickle.dump(patch, f)  # Save as pickle file in case of incorrect translation to image
    patch_np = patch.cpu().numpy().transpose(1, 2, 0)
    patch_image = Image.fromarray((patch_np * 255).astype(np.uint8))
    patch_image.save("adversarial_patch.png")
    print("Adversarial patch created and saved as 'adversarial_patch.png'")
