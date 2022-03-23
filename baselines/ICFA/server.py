import copy
import torch
from torch import nn
from torch.autograd import Variable
from data_and_model import models

class server(object):
    def __init__(self,args,device,clients_model):
        self.models = []
        self.init_model = models.get_model(args.dataset).to(device)
        self.args = args
        self.index = [[],[],[]]             # there are 3 clusters
        self.device = device
        self.clients_model = clients_model          # clients' models
        self.__initial_model()

    def __initial_model(self):              # initialize three models
        for i in range(3):
            model = models.get_model(self.args.dataset)
            self.models.append(model.to(self.device))
        

    def cluster_identify(self):
        # cluster identity and send model to client,respectively
        self.index = [[], [], []]       # clean indexes
        for i,client in enumerate(self.clients_model):
            losses = []
            for j in range(3):
                losses.append(self.test_loss(client,j))

            inde = torch.Tensor(losses).argmin().item()
            self.index[inde].append(i)

            self.clients_model[i].model.load_state_dict(self.models[inde].state_dict())
        
        print(self.index)


    def test_loss(self,client,j):
        model = self.models[j]
        model.eval()

        test_data = client.train_data
        loss = 0

        with torch.no_grad():
            for data,target in test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = model(data)
                #acc += torch.sum(torch.argmax(output,dim=1)==target).item()
                
                loss += nn.CrossEntropyLoss()(output, target).item()
                
        return loss



    
    def aggregate_model(self):
        # self.models = []
        for i,model in enumerate(self.models):
            m = copy.deepcopy(model.state_dict())

            if len(self.index[i]) == 0:
                continue

            for key in m.keys():
                m[key] = m[key] * 0.0
                for client in self.index[i]:
                    m[key] += self.clients_model[client].model.state_dict()[key]
                m[key] = torch.div(m[key],len(self.index[i]))
            model.load_state_dict(m)



    def gain_acc(self):
        
        test_correct = 0
        test_loss = 0

        for i in range(3):
            if len(self.index[i]) == 0:
                continue
            
            for j in self.index[i]:
                loss,acc = self.clients_model[j].test_model(self.models[i])
                test_loss += loss
                test_correct += acc

        test_acc = test_correct / self.args.global_nums

        return test_loss,test_acc


    def local_train(self):
        for client in self.clients_model:
            client.local_train()

