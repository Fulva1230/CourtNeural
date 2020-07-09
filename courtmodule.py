import torch

class CourtNet(torch.nn.Module):
    def __init__(self, jurors, num_of_layer, class_num):
        super(CourtNet, self).__init__()
        self.jurors = jurors
        self.class_num = class_num

        input_size = class_num * len(jurors)
        modules = []
        for i in range(num_of_layer):
            output_size = max(int(input_size // 1.1), class_num)
            modules.append(torch.nn.Linear(input_size, output_size))
            modules.append(torch.nn.ReLU)
            input_size = output_size

        modules.append(torch.nn.Linear(input_size, class_num))
        modules.append(torch.nn.Softmax(dim=1))
        self.sequenceModule = torch.nn.Sequential(*modules)


    def forward(self, x):
        jurors_pred = torch.cat([juror(x) for juror in self.jurors], 1)
        return self.sequenceModule(jurors_pred)