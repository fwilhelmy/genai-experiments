import matplotlib.pyplot as plt

def plot_intermediate_samples(images, steps_to_show, n_samples, n_steps):
    """
    Plot the intermediate steps of the diffusion process
    Args:
        images: List of image tensors at different steps
        steps_to_show: List of steps that were captured
        n_samples: Number of images to show
    """
    # Create a figure with n_samples rows and len(steps_to_show) columns
    plt.figure(figsize=(25, 15*n_samples))
    fig, axs = plt.subplots(n_samples, len(steps_to_show))
    # Plot each image
    for sample_idx in range(n_samples):
        for step_idx, img in enumerate(images):
            axs[sample_idx, step_idx].imshow(img[sample_idx, 0], cmap='gray')
            step = steps_to_show[step_idx] if step_idx < len(steps_to_show) else n_steps
            axs[sample_idx, step_idx].set_title(f' Image {sample_idx} \nt={n_steps - step-1}',size=8)
            axs[sample_idx, step_idx].axis('off')

    plt.tight_layout()
    plt.show()