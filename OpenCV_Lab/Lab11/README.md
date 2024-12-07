# Lab 11: Generative AI

This project demonstrates key concepts in Generative AI, including adversarial images, diffusion models, and generative adversarial networks (GANs).

## Features

### Adversarial Images

- Generates adversarial images by adding random noise to a grayscale image.
- Uses the `cv2.addWeighted` function for blending the original image with noise.
- Saves the adversarial image to disk and displays it.

### Additional Information

Adversarial images are inputs to machine learning models that are intentionally designed to cause the model to make a mistake. In the context of deep learning and computer vision, small perturbations to the input image, often imperceptible to humans, can cause a neural network to misclassify the image.

- **Fast Gradient Sign Method (FGSM)**: A popular technique for generating adversarial examples.

### Applications

- **Testing Model Robustness**: Evaluating how susceptible models are to adversarial attacks.
- **Improving Security**: Developing methods to defend against adversarial attacks.
- **Understanding Model Behavior**: Gaining insights into how models interpret input data.

### Diffusion Models

- Simulates adding Gaussian noise to an image in multiple steps (forward diffusion).
- Implements denoising using a Gaussian blur to reverse the added noise.
- Saves intermediate noisy images and the final denoised image to disk.

### Additional Information

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. They have recently gained popularity due to their ability to generate high-quality images.

- **Forward Process**: Gradually adds noise to data over multiple steps.
- **Reverse Process**: Trains a model to denoise and recover the original data.

### Applications

- **Image Generation**: Generating high-fidelity images from noise.
- **Inpainting**: Filling in missing parts of an image.
- **Super-Resolution**: Enhancing the resolution of images.

### Generative Adversarial Networks (GANs)

- Builds a simple GAN architecture with:
  - **Generator**: Creates fake images from random noise.
  - **Discriminator**: Classifies images as real or fake.
- Demonstrates the architecture of both models but does not train them in this script.

### Additional Information

GANs are a class of machine learning frameworks where two neural networks compete with each other in a game. They have been widely used for generating realistic images, videos, and audio.

- **Generator (G)**: Attempts to produce data that is indistinguishable from real data.
- **Discriminator (D)**: Attempts to distinguish between real data and data produced by the generator.

### Training Process

- The generator and discriminator are trained simultaneously.
- The generator tries to minimize the discriminator's ability to correctly classify fake data.
- The discriminator tries to maximize its accuracy.

### Challenges with GANs

- **Training Instability**: GANs can be difficult to train due to issues like mode collapse.
- **Hyperparameter Tuning**: Requires careful selection of learning rates and architectures.
- **Evaluation Metrics**: Measuring the quality of generated images is non-trivial.

### Applications of GANs

- **Image Generation**: Creating realistic images from random noise.
- **Style Transfer**: Transferring styles between images.
- **Data Augmentation**: Generating synthetic data to augment training datasets.
- **Super-Resolution**: Enhancing image resolution.

---

## Function Details and Parameters

### `create_adversarial_image(image_path, save_path)`

Generates and saves an adversarial image by adding random noise.

- **Parameters**:
  - `image_path` (str): Path to the input image file.
  - `save_path` (str): Path to save the generated adversarial image.

### Code Snippet

```python
def create_adversarial_image(image_path, save_path):
    image = cv2.imread(image_path, 0)  # Load image in grayscale
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    adversarial_image = cv2.addWeighted(image, 0.5, noise, 0.5, 0)
    cv2.imwrite(save_path, adversarial_image)
```

### `apply_diffusion(image_path, save_folder)`

Simulates a diffusion process by adding Gaussian noise in steps and saving the noisy images.

- **Parameters**:
  - `image_path` (str): Path to the input image file.
  - `save_folder` (str): Path to the folder where intermediate noisy images will be saved.

### Code Snippet

```python
def apply_diffusion(image_path, save_folder):
    image = cv2.imread(image_path, 0)
    num_steps = 10
    noisy_image = image.copy()
    for i in range(num_steps):
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy_image = cv2.add(noisy_image, noise)
        cv2.imwrite(f"{save_folder}/noisy_step_{i}.png", noisy_image)
```

### `denoise_image(noisy_image_path, save_path)`

Applies Gaussian blur to denoise a noisy image and saves the result.

- **Parameters**:
  - `noisy_image_path` (str): Path to the input noisy image file.
  - `save_path` (str): Path to save the denoised image.

### Code Snippet

```python
def denoise_image(noisy_image_path, save_path):
    noisy_image = cv2.imread(noisy_image_path, 0)
    denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
    cv2.imwrite(save_path, denoised_image)
```

### GAN Classes

### `Generator(latent_dim)`

- **Parameters**:
  - `latent_dim` (int): Dimension of the input noise vector.
- **Description**: Creates fake images from random noise using a neural network.

### `Discriminator`

- **Description**: Classifies images as real or fake using a neural network.

### Code Snippet

```python
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(-1, 28*28))
```

---

## Prerequisites

- Python 3.8+
- Libraries:
  - OpenCV
  - NumPy
  - PyTorch

Install the required libraries using:

```bash
pip install numpy opencv-python torch torchvision
```

## How to Run

1. **Prepare an Image**:

   Place a grayscale image file named `image.jpg` in the same directory as the script.

2. **Install Dependencies**:

   ```bash
   pip install numpy opencv-python torch torchvision
   ```

3. **Run the Script**:

   ```bash
   python3 lab11.py
   ```

## References

- **Generative AI Applications**: Used in digital art, content creation, and medical imaging.
- **Diffusion Models**: As alternatives to GANs for handling complex data distributions.
- **FGSM Technique**: For adversarial image generation.
- **GAN Research**: Papers by Ian Goodfellow and others on generative adversarial networks.
- **Denoising Diffusion Probabilistic Models**: Papers by Ho et al. on diffusion models.

### Additional Resources

- **PyTorch Tutorials**: For implementing and training GANs.
- **OpenAI's DALL·E**: An example of a powerful generative model.
- **Adversarial Attacks and Defenses**: Research on improving model robustness.
