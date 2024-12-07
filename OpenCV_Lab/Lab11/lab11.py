import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# === Adversarial Images ===
def create_adversarial_image(image_path, save_path):
    # Load the original image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    # Create noise with a standard deviation of 25
    noise = np.random.normal(0, 25, image.shape).astype('uint8')

    # Combine original image with noise
    adversarial_image = cv2.addWeighted(image, 1.0, noise, 0.1, 0)
    
    # Save and display the adversarial image
    cv2.imwrite(save_path, adversarial_image)
    cv2.imshow("Adversarial Image", adversarial_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Diffusion Models ===
def apply_diffusion(image_path, save_folder):
    # Ensure the folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found. Please check the path.")
    
    image = cv2.resize(image, (128, 128)) / 255.0  # Normalize to [0, 1]
    
    # Add Gaussian noise in steps
    for i in range(5):
        noise = np.random.normal(0, 0.1 * (i + 1), image.shape)
        noisy_image = np.clip(image + noise, 0, 1)  # Clip values to [0, 1]
        noisy_image_path = f"{save_folder}/noisy_step_{i+1}.png"
        cv2.imwrite(noisy_image_path, (noisy_image * 255).astype('uint8'))
        cv2.imshow(f"Step {i+1}", (noisy_image * 255).astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def denoise_image(noisy_image_path, save_path):
    # Load the noisy image
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE) / 255.0  # Normalize
    if noisy_image is None:
        raise FileNotFoundError("Noisy image not found. Please check the path.")
    
    # Apply Gaussian blur to denoise
    denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
    cv2.imwrite(save_path, (denoised_image * 255).astype('uint8'))
    cv2.imshow("Denoised Image", (denoised_image * 255).astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Generative Adversarial Networks (GANs) ===
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Example of model instantiation
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()

# === Main Execution ===
if __name__ == "__main__":
    image_path = "image.jpg"

    # Adversarial Image Generation
    create_adversarial_image(image_path, "adversarial_image.png")

    # Diffusion Model - Forward Process
    apply_diffusion(image_path, "diffusion_steps")

    # Diffusion Model - Denoising
    denoise_image("diffusion_steps/noisy_step_5.png", "denoised_image.png")
