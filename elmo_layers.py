import torch
from torch import nn


class CharCNN(nn.Module):

    def __init__(self,out_dim=128,char_dim=16):

        super().__init__()

        self.out_dim = out_dim
        self.char_emb = nn.Embedding(262,char_dim)

        conv_filters = (
        (1,32),(2,32),(3,64),(4,128),(5,256),(6,512),(7,1024)
        )

        self.cnn_layers = nn.ModuleList([nn.Conv1d(char_dim,
                                                   n_filters,
                                                   window,
                                                   padding=0)
                                         for window,n_filters in conv_filters])

        self.gate = nn.Linear(2048,2048)
        self.lin_trans = nn.Linear(2048,2048)
        self.out = nn.Linear(2048,out_dim)



    def forward(self,x):

        batch,maxlen,chars = x.shape

        # char embedding
        x = x.reshape((batch * maxlen,chars))
        x = self.char_emb(x)
        x = x.transpose(1,2)

        # char cnn
        cnn_out = []

        for conv in self.cnn_layers:

            out = torch.relu(conv(x))
            out,_ = torch.max(out,2)
            cnn_out.append(out)

        cnn_out = torch.concat(cnn_out,dim=-1)

        # highway connection
        lin_trans = torch.relu(self.lin_trans(cnn_out))
        gate = torch.sigmoid(self.gate(cnn_out))
        x = gate * lin_trans + (1 - gate) * cnn_out

        # output projection
        x = self.out(x)
        x = x.reshape((batch,maxlen,self.out_dim))

        return x


class LSTMELMo(nn.Module):

    def __init__(self,units=1024,proj=128):

        super().__init__()

        self.units = units
        self.proj = proj

        self.lstm0 = nn.LSTM(input_size=128, hidden_size=units, batch_first=True,
                             proj_size=proj, bidirectional=True)

        self.lstm1 = nn.LSTM(input_size=256, hidden_size=units, batch_first=True,
                             proj_size=proj, bidirectional=True,)

        self.lstm0.requires_grad_(requires_grad=False)
        self.lstm1.requires_grad_(requires_grad=False)


        self.k = nn.Parameter(torch.empty(3,))  # Default to [1, 1, 1]
        # self.g = nn.Parameter(torch.empty(1,))

        nn.init.constant_(self.k,1)
        # nn.init.constant_(self.g, 1)


        for lstm in [self.lstm0, self.lstm1]:
            lstm.bias_hh_l0.data.zero_()
            lstm.bias_hh_l0_reverse.data.zero_()
            lstm.bias_hh_l0.requires_grad = False
            lstm.bias_hh_l0_reverse.requires_grad = False


    def forward(self,x):

        lstm0,_ = self.lstm0(x)
        lstm1,_ = self.lstm1(lstm0)
        lstm1 = lstm1 + lstm0

        k = torch.softmax(self.k,dim=0)

        elmo_f = k[0] * x + k[1] * lstm0[:,:,:self.proj] + k[2] * lstm1[:,:,:self.proj]
        elmo_b = k[0] * x + k[1] * lstm0[:,:,self.proj:] + k[2] * lstm1[:,:,self.proj:]

        elmo = torch.concat([elmo_f,elmo_b],-1)

        return elmo


class ELMoLayer(nn.Module):

    def __init__(self,):
        super().__init__()
        self.char_cnn = CharCNN()
        self.char_cnn.requires_grad_(requires_grad=False)
        self.lstm_elmo = LSTMELMo()

    def forward(self,x):

        cnn = self.char_cnn(x)
        elmo = self.lstm_elmo(cnn)
        return elmo





















