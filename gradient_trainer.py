import torch
import numpy as np

def train_epoch(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion):
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    gradients = []
    model = model.to(device=device)
    model.train()  # Ensure the model is in training mode

    for index, (x, y) in enumerate(unlabel_loader):
        x = x.to(device=device, dtype=dtype)
        pseudo_y = pseudo_labels[index * batch_size: index * batch_size + batch_size].to(device=device, dtype=torch.int64)

        # Process input through the model up to the penultimate layer
        penultimate_output = model.forward_to_penultimate(x)

        # Detach the output of the penultimate layer from the computation graph
        penultimate_output_detached = penultimate_output.detach()
        penultimate_output_detached.requires_grad = True  # Enable gradient computation for this tensor

        scores = model.part2(penultimate_output_detached)
        loss = criterion(scores, pseudo_y)
        loss.backward()

        if penultimate_output_detached.grad is not None:
            # Convert the gradient to a NumPy array and store it
            gradients.append(penultimate_output_detached.grad.cpu().detach().numpy())
        else:
            gradients.append(None)

    print("Gradients Computation DONE!")
    gradients = np.concatenate(gradients, axis=0)
    return gradients