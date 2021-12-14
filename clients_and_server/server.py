from data_and_model.models import get_model
import copy
import torch
from torch import nn
from torch.autograd import Variable


class get_server(object):
    def __init__(self,model,device,clients,args):
        # self.model = copy.deepcopy(get_model(dataset=args.dataset).to(device))
        self.model = copy.deepcopy(model.to(device))
        self.device = device
        self.clients = clients
        self.args = args


    def get_cluster_model(self,clients_id):
        client_models = []
        for i in range(self.args.global_nums):
            client_model = torch.load('./cache/client_model_{}.pt'.format(i))
            client_models.append(copy.deepcopy(client_model))

        # get k cluster models
        self.cluster_models = [copy.deepcopy(self.model.state_dict()) for _ in range(len(clients_id))]

        for i,client_id in enumerate(clients_id):
            # create cluster models

            # model = copy.deepcopy(self.cluster_models[i].state_dict())
            for key in self.cluster_models[i].keys():
                self.cluster_models[i][key] = self.cluster_models[i][key] * 0.0
                for id in client_id:
                    self.cluster_models[i][key] += client_models[id][key]
                self.cluster_models[i][key] = torch.true_divide(self.cluster_models[i][key],len(client_id))
            torch.save(self.cluster_models[i], './cache/cluster_model_{}.pt'.format(i))

    
    
    # avg cluster models and send to clients
    def get_global_model(self):
        # self.global_model = copy.deepcopy(self.model)
        param = copy.deepcopy(self.model.state_dict())

        for key in param.keys():
            param[key] = param[key]*0
            for model in self.cluster_models:
                param[key] = param[key] + model[key]
            param[key] = torch.true_divide(param[key],len(self.cluster_models))
            # param[key] = param[key]/len(self.cluster_models)
        torch.save(param, './cache/global_model.pt')
        # self.global_model.load_state_dict(param)



    def client_get_model(self,clients_id):
        pass
