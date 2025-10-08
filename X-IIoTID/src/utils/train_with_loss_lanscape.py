import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def get_random_directions(params):
    """
    Generate two random normalized directions matching parameter shapes.

    Random direction are chosen in order to avoid exploration bias

    Sampling on layer basis ensures the random direction is balanced relative to the contribution of each layer
    """
    d1, d2 = [], []
    for p in params:
        r1 = torch.randn_like(p)
        r2 = torch.randn_like(p)

        # Normalize directions relative to parameter norm so that they are comaprable to modle paraùs
        r1 = r1 / (torch.norm(r1) + 1e-10) * torch.norm(p)
        r2 = r2 / (torch.norm(r2) + 1e-10) * torch.norm(p)

        d1.append(r1)
        d2.append(r2)
    return d1, d2


def perturb_model(model, original_state, d1, d2, alpha, beta):
    """
    Return a perturbed state_dict by moving along directions d1, d2.
    Only applies perturbations to trainable parameters.
    """
    new_state = copy.deepcopy(original_state)
    param_names = [name for name, _ in model.named_parameters()]

    for name, dd1, dd2 in zip(param_names, d1, d2):
        new_state[name] = original_state[name] + alpha * dd1 + beta * dd2

    return new_state


def plot_loss_landscape_2d(
    model,
    criterion,
    data_loader,
    device,
    epoch,
    param_range=0.5,
    resolution=30,
    is_binary=True,
    smoothing_sigma=0.8,
):
    """
    Plot a smoother loss landscape with continuous values.

    Args:
        smoothing_sigma: Sigma parameter for Gaussian smoothing (higher = smoother)
    """
    model.eval()

    # Store original parameters
    original_state = copy.deepcopy(model.state_dict())
    params = [p.clone().detach() for p in model.parameters()]

    # Random directions
    d1, d2 = get_random_directions(params)

    # Create a finer grid for smoother visualization
    alphas = np.linspace(-param_range, param_range, resolution)
    betas = np.linspace(-param_range, param_range, resolution)
    A, B = np.meshgrid(alphas, betas)

    losses = np.zeros((resolution, resolution))

    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                new_state = perturb_model(
                    model, original_state, d1, d2, A[i, j], B[i, j]
                )
                model.load_state_dict(new_state)

                total_loss, batch_count = 0, 0
                # Use more batches for better loss estimation
                for inputs, targets in data_loader:
                    # targets = targets.float()
                    if batch_count >= 10:  # Increased from 5 to 10
                        break
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    if is_binary:
                        targets = targets.view(-1, 1).float()
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    batch_count += 1
                losses[i, j] = total_loss / batch_count

    # Restore original parameters
    model.load_state_dict(original_state)

    # Apply Gaussian smoothing to make the landscape smoother
    if smoothing_sigma > 0:
        losses = gaussian_filter(losses, sigma=smoothing_sigma)

    # Create an even finer grid for interpolation
    fine_resolution = 100
    alpha_fine = np.linspace(-param_range, param_range, fine_resolution)
    beta_fine = np.linspace(-param_range, param_range, fine_resolution)
    A_fine, B_fine = np.meshgrid(alpha_fine, beta_fine)

    # Flatten the original grid for interpolation
    points = np.column_stack((A.ravel(), B.ravel()))
    values = losses.ravel()

    # Interpolate to finer grid using cubic interpolation
    losses_fine = griddata(
        points, values, (A_fine, B_fine), method="cubic", fill_value=values.mean()
    )

    # Compute current (unperturbed) loss
    current_loss, batch_count = 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            # targets = targets.float()
            if batch_count >= 10:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if is_binary:
                targets = targets.view(-1, 1).float()
            loss = criterion(outputs, targets)
            current_loss += loss.item()
            batch_count += 1
    current_loss /= batch_count

    # Plotting with finer, smoother surface
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Use the finer grid for plotting
    surf = ax.plot_surface(
        A_fine,
        B_fine,
        losses_fine,
        cmap="viridis",
        alpha=0.9,
        antialiased=True,
        linewidth=0,
        rstride=1,
        cstride=1,
    )

    ax.scatter([0], [0], [current_loss], color="red", s=100, label="Current Position")

    ax.set_xlabel("α (Direction 1)")
    ax.set_ylabel("β (Direction 2)")
    ax.set_zlabel("Loss")
    ax.set_title(f"Smooth Loss Landscape - Epoch {epoch}")
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Loss Value")

    output_dir = "/out/figs/metric_distributions"
    plt.tight_layout()
    plt.show()

    return losses_fine
