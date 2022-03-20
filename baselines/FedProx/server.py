import copy
import torch
from torch import nn
from torch.autograd import Variable


class server(object):
    def __init__(self,model,device,dataset,clients_id):
        self.model = model.to(device)
        self.device = device
        self.clients_id = clients_id      # 这个簇的用户集合
        self.test_data = dataset


    # 普通的FedAvg用到这个函数，基于聚类的FedAvg用下面的函数
    def aggregate_model(self):
        model_states = []
        for i in self.clients_id:
            model_states.append(torch.load('./cache/model_state_{}.pt'.format(i)))

        global_model_state = copy.deepcopy(model_states[0])

        for key in global_model_state.keys():
            global_model_state[key] = global_model_state[key]*0.0
            for i in range(0, len(model_states)):
                global_model_state[key] += model_states[i][key]
            global_model_state[key] = torch.div(global_model_state[key], len(model_states))

        self.model.load_state_dict(global_model_state)
        torch.save(self.model.state_dict(), './cache/global_model.pt')


    def gain_acc(self):
        
        test_correct = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = test_correct / len(self.test_data.dataset)
        
        return test_loss,test_acc



