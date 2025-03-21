import os
import torch
from utils.interaction_tensor import *


# Expect directiory as
# saved_dir/
#   - 1/
#       -epoch_1.pt
#       -epoch_2.pt
#   - 2/
#       -epoch_1.pt
#       -epoch_2.pt

def return_model_list(epoch, num_model, saved_dir): 
    model_list = []
    for i in range(1, num_model+1):
        client_dir = os.path.join(saved_dir, str(i))
        model_path = os.path.join(client_dir, f'epoch_{epoch}.pt')
        if os.path.isfile(model_path) == True:
            model = torch.load(model_path)
            if any(key.startswith("module.") for key in model.keys()):
                model = {key.replace("module.", ""): value for key, value in model.items()}
            model_list.append(model)
    assert(len(model_list) == num_model)

    return model_list


def feature_frequency(interaction_tensor): # Figure 2(a),(b)
    frequency_tensor = torch.sum(interaction_tensor, dim=0).bool().int()
    frequency_tensor = torch.sum(frequency_tensor, dim=0)
    frequency, _ = torch.sort(frequency_tensor)
    return frequency


def ndata_nmodel(interaction_tensor): # Figure 2(b)

    n_model = torch.sum(interaction_tensor, dim=1).bool().int()
    n_model = torch.sum(n_model, dim=0)

    n_data = torch.sum(interaction_tensor, dim=0).bool().int()
    n_data = torch.sum(n_data, dim=0)

    return n_data, n_model


"""
Not Implemented yet

def shared_error(): # Figure 3(c)

    m_feat = torch.sum(interaction_tensor, dim=1).bool().int()



def confidence_features(model, model_list, cluster): # Figure 3(a)

    for i, m in enumerate(model_list):

        model.load_state_dict(m)

        conf = []
        acc = []

        for j, (xs, ys) in enumerate(dataloader):

            xs.cuda()
            ys.cuda()

            logits = model(xs)
            conf, pred = logits.topk(1, 1, True, True)

            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))


        last_layer_output = torch.cat(last_layer_output, dim=0)
        _, _, V = torch.linalg.svd(last_layer_output)
        proj_output = torch.matmul(last_layer_output, V[:K].T)

"""