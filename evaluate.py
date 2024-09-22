import torch


def reconstruction_loss(x, x_rec):
    """
    :param x: the original images
    :param x_rec: the reconstructed images
    :return: the reconstruction loss
    """
    return torch.norm(x - x_rec) / torch.prod(torch.tensor(x.shape))


def evaluate_model(model, test_dl, opt, latents, epochs, device):
    """
    :param model: the trained model
    :param test_dl: a DataLoader of the test set
    :param opt: a torch.optim object that optimizes ONLY the test set
    :param latents: initial values for the latents of the test set
    :param epochs: how many epochs to train the test set latents for
    :return:
    """
    for epoch in range(epochs):
        for i, x in test_dl:
            i = i.to(device)
            x = x.to(device)
            x_rec = model(latents[i])
            loss = reconstruction_loss(x, x_rec)
            opt.zero_grad()
            loss.backward()
            opt.step()

    losses = []
    with torch.no_grad():
        for i, x in test_dl:
            i = i.to(device)
            x = x.to(device)
            x_rec = model(latents[i])
            loss = reconstruction_loss(x, x_rec)
            losses.append(loss.item())

        final_loss = sum(losses) / len(losses)

    return final_loss

