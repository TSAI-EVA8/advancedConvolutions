import torch
import torch
import torch.nn.functional as F
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

def set_seed(seed, cuda):
    """ Setting the seed makes the results reproducible. """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def initialize_cuda(seed):
    """ Check if GPU is availabe and set seed. """

    # Check CUDA availability
    cuda = torch.cuda.is_available()
    print('GPU Available?', cuda)

    # Initialize seed
    set_seed(seed, cuda)

    # Set device
    device = torch.device("cuda" if cuda else "cpu")

    return cuda,device 




def eval(model, loader, device, criterion, losses, accuracies, incorrect_samples):
    """Evaluate the model.
    Args:
        model: Model instance.
        loader: Validation data loader.
        device: Device where the data will be loaded.
        criterion: Loss function.
        losses: List containing the change in loss.
        accuracies: List containing the change in accuracy.
        incorrect_samples: List containing incorrectly predicted samples.
    """

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            img_batch = data  # This is done to keep data in CPU
            data, target = data.to(device), target.to(device)  # to gpu
            output = model(data)  # Get trained model output
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            result = pred.eq(target.view_as(pred))

            # Save incorrect samples
            if len(incorrect_samples) < 25:
                for i in range(loader.batch_size):
                    if not list(result)[i]:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                        })

            correct += result.sum().item()

    val_loss /= len(loader.dataset)
    losses.append(val_loss)
    accuracies.append(100. * correct / len(loader.dataset))

    print(
        f'\nTest set: Average loss: {val_loss:.4f}, Test Accuracy: {correct}/{len(loader.dataset)} ({accuracies[-1]:.2f}%)\n'
    )






def train(model, loader, device, optimizer, criterion, l1_factor=0.0):
    """Train the model.
    Args:
        model: Model instance.
        device: Device where the data will be loaded.
        loader: Training data loader.
        optimizer: Optimizer for the model.
        criterion: Loss Function.
        l1_factor: L1 regularization factor.
    """

    model.train()
    pbar = tqdm(loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar, 0):
        # Get samples
        data, target = data.to(device), target.to(device) # move to the GPU

        # Set gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Predict output
        y_pred = model(data)

        # Calculate loss
        loss = l1(model, criterion(y_pred, target), l1_factor)

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Update Progress Bar
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Accuracy={(100 * correct / processed):.2f}'
        )


def l1(model, loss, Lambda):
    """Apply L1 regularization.
    Args:
        model: Model instance.
        loss: Loss function value.
        factor: Factor for applying L1 regularization
    
    Returns:
        Regularized loss value.
    """

    if Lambda > 0:
        criteria = nn.L1Loss(size_average=False)
        regularizer_loss = 0
        for parameter in model.parameters():
            regularizer_loss += criteria(parameter, torch.zeros_like(parameter))
        loss += Lambda * regularizer_loss
    return loss


def cross_entropy_loss():
    """Create Cross Entropy Loss
    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()


def sgd_optimizer(model, learning_rate, momentum, l2_factor=0.0):
    """Create optimizer.
    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum of optimizer.
        l2_factor: Factor for L2 regularization.
    
    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_factor
    )




