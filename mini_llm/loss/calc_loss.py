import torch.nn.functional as F

def calc_loss_batch(input_batch, target_batch, model, device):
    # device to allow us to transfer to a given device like GPU
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # sum the loss over each batch
            total_loss += loss.item()
        else:
            break

    # avg the loss over all batches
    return total_loss / num_batches
