import torch

class CourtNet(torch.nn.Module):
    def __init__(self, jurors, class_num):
        super(CourtNet, self).__init__()
        self.jurors = jurors
        self.class_num = class_num
        l1_output_size = class_num * len(jurors) // 2
        self.l1 = torch.nn.Linear(class_num * len(jurors), l1_output_size)
        self.f1 = torch.nn.functional.relu
        self.l2 = torch.nn.Linear(l1_output_size, class_num)
        self.params1 = self.l1.parameters()
        self.params2 = self.l2.parameters()
        def f2(input_tensor):
            return torch.nn.functional.softmax(input_tensor, 1)
        self.f2 = f2

    def forward(self, x):
        jurors_pred = torch.cat([juror(x) for juror in self.jurors], 1)
        return self.f2(self.l2(self.f1(self.l1(jurors_pred))))