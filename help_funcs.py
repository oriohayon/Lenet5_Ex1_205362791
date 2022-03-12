# def helper functions
import torch


@torch.no_grad()
def get_all_preds(data_loader, model):
    """function to return all predicted probabilities"""
    all_preds = torch.tensor([])  # init empty tensor
    for batch in data_loader:
        imgs, lbls = batch
        preds = model(imgs)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


def get_correct(preds, lbls):
    """function tells us how many predictions are correct"""
    return preds.argmax(1).eq(lbls).sum().item()