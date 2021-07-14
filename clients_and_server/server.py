import copy
import torch
from torch.autograd import Variable


class server(object):
    def __init__(self,model,device,dataset,clients_id,server_id):
        self.model = model.to(device)
        self.device = device
        self.clients_id = clients_id      # 这个簇的用户集合
        self.server_id = server_id        # 这个簇的编号
        self.acc = []
        self.test_data = dataset[1]


    # 普通的FedAvg用到这个函数，基于聚类的FedAvg用下面的函数
    def aggregate_model(self):
        model_states = []
        for i in self.clients_id:
            model_states.append(torch.load('./cache/model_state_{}.pt'.format(i)))

        global_model_state = copy.deepcopy(model_states[0])

        for key in global_model_state.keys():
            for i in range(1, len(model_states)):
                global_model_state[key] += model_states[i][key]
            global_model_state[key] = torch.div(global_model_state[key], len(model_states))

        self.model.load_state_dict(global_model_state)
        torch.save(self.model.state_dict(), './cache/global_model_{}.pt'.format(self.server_id))


    def gain_acc(self):
        
        test_correct = 0
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = test_correct / len(self.test_data.dataset)
        #print('[Global model]  test_accuracy: {:.3f}%'.format(test_acc * 100.))
        return test_acc


    # 这个函数用于聚合所有的簇
    def aggregate_cluster(self):
        model_states = []
        for i in self.clients_id:
            model_states.append(torch.load('./cache/global_model_{}.pt'.format(i)))

        global_model_state = copy.deepcopy(model_states[0])

        for key in global_model_state.keys():
            for i in range(1, len(model_states)):
                global_model_state[key] += model_states[i][key]
            global_model_state[key] = torch.div(global_model_state[key], len(model_states))

        self.model.load_state_dict(global_model_state)
        torch.save(self.model.state_dict(), './cache/global_model.pt')

