import torch

class CourtNet(torch.nn.Module):
    def __init__(self, jurors, class_num):
        super(CourtNet, self).__init__()
        self.jurors = jurors
        self.class_num = class_num
        l1_output_size = class_num * len(jurors)
        self.l1 = torch.nn.Linear(class_num * len(jurors), l1_output_size)
        self.f1 = torch.nn.functional.relu
        l2_output_size = max(int(l1_output_size // 1.2), class_num)
        self.l2 = torch.nn.Linear(l1_output_size, l2_output_size)
        self.f2 = torch.nn.functional.relu

        l3_output_size = max(int(l2_output_size // 1.1), class_num)
        self.l3 = torch.nn.Linear(l2_output_size, l3_output_size)

        self.f3 = torch.nn.functional.leaky_relu
        self.l4 = torch.nn.Linear(l3_output_size, class_num)
        def finalf(input_tensor):
            return torch.nn.functional.softmax(input_tensor, 1)
        self.finalf = finalf
        self.params1 = self.l1.parameters()
        self.params2 = self.l2.parameters()
        self.params3 = self.l3.parameters()
        self.params4 = self.l4.parameters()


    def forward(self, x):
        jurors_pred = torch.cat([juror(x) for juror in self.jurors], 1)
        return self.finalf(self.l4(self.f3(self.l3(self.f2(self.l2(self.f1(self.l1(jurors_pred))))))))