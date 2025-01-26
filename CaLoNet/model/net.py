from model.layer import *
import torch
import torch.nn as nn
from model.attention import *
from models.layer import DeGINConv, DenseGCNConv, DenseSAGEConv, DenseGraphConv, rkGraphConv
from model.layer import printt
class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None,
                 static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1,  # 2

                 conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128,  # 16,16,32,64
                 # 序列长度，输入维度1，输出维度1，
                 seq_length=12, in_dim=2, out_dim=12, layers=3,
                 propalpha=0.05, tanhalpha=3, layer_norm_affline=True,

                 # t=1, num_classes=1, channels=1  # 随便设置的
                 # , hidden_dim=(96, 192, 62),  #
                 t=1,  # 长度
                 down_dim=1024,  # length = 1536 * 2，降维维度

                 hidden_dim=(96, 62),  ##192
                 layers1=(2, 2, 6, 2),
                 heads=(3, 6, 12, 24),
                 channels=1,
                 num_classes=1,
                 head_dim=32,
                 window_size=1,
                 downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
                 relative_pos_embedding=True,
                 wa=1,
                 prob=1,
                 mask=1,

                 ):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.a0 = 3
        self.a1 = 5
        self.a2 = 7
        self.a3 = self.a0 + self.a1 + self.a2
        self.reduction = 32
        self.multiscale = nn.ModuleList()
        # self.filter_convs = nn.ModuleList()  ##
        # 如果你想设计一个神经网络的层数作为输入传递，当添加 nn.ModuleList
        # 作为 nn.Module 对象的一个成员时（即当我们添加模块到我们的网络时），
        # 所有 nn.ModuleList 内部的 nn.Module 的 parameter 也被添加作为 我们的网络的 parameter。
        # self.gate_convs = nn.ModuleList()
        # self.residual_convs = nn.ModuleList()
        # self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.SELayer = nn.ModuleList()
        # self.norm = nn.ModuleList()  ##标准化
        # self.Diffpool1 = DiffPool(32, 32, gcn_depth, dropout, propalpha, device,
        #                           final_layer=False)
        # self.Diffpool2 = DiffPool(32, 1, gcn_depth, dropout, propalpha, device,
        #                           final_layer=True)

        ##=========
        self.start_conv = nn.Conv2d(in_channels=in_dim,  ##[N, C, H, W]中的C
                                    out_channels=residual_channels,  ##16
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)
        ##邻接矩阵

        self.seq_length = seq_length  # 输入序列
        kernel_size = 7
        # if dilation_exponential>1:##默认2
        #     self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        # else:
        #     self.receptive_field = layers*(kernel_size-1) + 1
        #
        # for i in range(1):
        #     if dilation_exponential>1:
        #         rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        #     else:
        #         rf_size_i = i*layers*(kernel_size-1)+1
        #     new_dilation = 1
        #     for j in range(1,layers+1):
        #         if dilation_exponential > 1:
        #             rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
        #         else:
        #             rf_size_j = rf_size_i+j*(kernel_size-1)
        #
        #         self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
        #         #layer调用，nn.moduleList定义对象后，有extend和append方法，用法和python中一样，
        #         # extend是添加另一个modulelist  append是添加另一个module
        #         self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
        #         self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
        #                                             out_channels=residual_channels,
        #                                          kernel_size=(1, 1)))
        # if self.seq_length>self.receptive_field:
        #     self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
        #                                     out_channels=skip_channels,
        #                                     kernel_size=(1, self.seq_length-rf_size_j+1)))
        # else:
        #     self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
        #                                     out_channels=skip_channels,
        #                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

        # 图卷积
        # if self.gcn_true:  #
        # print('1')
        ##16,16，gcn,0.05
        #diffpool
        # self.gconv1.append(
        #     mixprop(conv_channels, 32, gcn_depth, dropout, propalpha))  # conv_channels, residual_channels
        # self.gconv1.append(mixprop(32, 32, gcn_depth, dropout, propalpha))
        #
        # self.gconv2.append(mixprop(conv_channels, 32, gcn_depth, dropout, propalpha))
        # self.gconv2.append(mixprop(32, 32, gcn_depth, dropout, propalpha))

        # self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
        #
        # self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

        # if self.seq_length > self.receptive_field:
        #
        #     #
        #     self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
        #                                elementwise_affine=layer_norm_affline))
        # else:
        #     self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
        #                                elementwise_affine=layer_norm_affline))

        # new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=conv_channels,  # 32,skip_channels
                                    out_channels=end_channels,  # 64
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,  # 64
                                    out_channels=out_dim,  # 1
                                    kernel_size=(1, 1),
                                    bias=True)
        # if self.seq_length > self.receptive_field:
        #     self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
        #     #
        #     self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        #
        # else:
        #     self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
        #     #
        #     self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)#

        self.idx = torch.arange(self.num_nodes).to(device)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(3072),  ##62,normalized_shape：归一化的维度，int（最后一维）list（list里面的维度）#hidden_dim[-1]
            nn.Linear(3072, num_classes)  # hidden_dim[-1]
        )

        self.multiscale = nn.ModuleList([
            nn.Conv1d(in_channels=num_nodes, out_channels=conv_channels, kernel_size=self.a0, padding=1),
            nn.Conv1d(in_channels=num_nodes, out_channels=conv_channels, kernel_size=self.a1, padding=2),
            nn.Conv1d(in_channels=num_nodes, out_channels=conv_channels, kernel_size=self.a2, padding=3)

        ]
        )

        self.SELayer.append(SELayer(48, self.reduction))  ##超参数

        self.attention = nn.ModuleList()
        self.attention.append(attentionNet(
            t=t,  # 长度
            down_dim=down_dim,  # length = 1536 * 2，降维维度
            hidden_dim=hidden_dim,
            layers=layers1,

            heads=heads,
            channels=channels,
            num_classes=num_classes,
            head_dim=head_dim,
            window_size=window_size,
            downscaling_factors=downscaling_factors,  # 代表多长的时间作为一个特征
            wa=wa,
            prob=prob,
            mask=mask,

        )).to(device)
        # self.attention.append(attentionNet(
        #     t=t,  # 长度
        #     down_dim=down_dim,  # length = 1536 * 2，降维维度
        #     hidden_dim=(96, 62),
        #     layers=(2, 2, 6, 2),
        #
        #     heads=(3, 6, 12, 24),
        #     channels=channels,
        #     num_classes=num_classes,
        #     head_dim=32,
        #     window_size=window_size,
        #     downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
        #
        #     relative_pos_embedding=True,
        #     wa=wa,
        #     prob=prob,
        #     mask=mask,
        #     k=32,
        #     dim_head=None,
        #     one_kv_head=True,
        #     share_kv=False
        # ))

        self.gate = torch.nn.Linear(num_nodes * t + 768 * t, 2)  # x y z + 48*t
        # self.gate = torch.nn.Linear(num_nodes + 48 + 768 , 3)  # x y z
        self.output_linear = nn.Sequential(
            nn.LayerNorm(num_nodes * t + 768 * t),  ##62,normalized_shape：归一化的维度，int（最后一维）list（list里面的维度）#hidden_dim[-1]
            torch.nn.Linear(num_nodes * t + 768 * t, num_classes)
        )
        # self.output_linear = torch.nn.Linear(num_nodes*t + 48*t + 768*t, num_classes)

        self.cbamx = cbamblock1(16, 16, 7)
        self.cbamz = cbamblock1(768, 16, 7)

        self.decoder = 'GIN'#GIN最好用 rGNN SAGE GCN
        #basicmotios GIN
        hid1=3072
        hid2=3072
        if self.decoder == 'GNN':
            # 多
            # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(3072, 3072)

            self.gnn2 = DenseGraphConv(3072, 3072)
            self.gnn3 = DenseGraphConv(3072, 3072)
        if self.decoder == 'GCN':
            # 多
            self.gcn1 = DenseGCNConv(hid1, hid1)
            self.gcn2 = DenseGCNConv(hid1, hid2)
            self.gcn3 = DenseGCNConv(hid2, hid2)
        if self.decoder == 'SAGE':
            # 多
            self.sage1 = DenseSAGEConv(hid1, hid1)#(d, args.hid1)
            self.sage2 = DenseSAGEConv(hid1, hid1)#(args.hid1, args.hid2)
            self.sage3 = DenseSAGEConv(hid1, hid1)#(args.hid2, args.hid2)
        if self.decoder == 'rGNN':
            #多
            self.num_adjs = 1
            self.attention_mode = 'naive'
            self.gc1 = rkGraphConv(self.num_adjs, hid1, hid1, self.attention_mode, aggr='mean')
            self.gc2 = rkGraphConv(self.num_adjs, hid1, hid1, self.attention_mode, aggr='mean')
            self.gc3 = rkGraphConv(self.num_adjs, hid1, hid1, self.attention_mode, aggr='mean')

        if self.decoder == 'GIN':
            # ginnn = nn.Sequential(
            #     nn.Linear(d, args.hid1),
            #     # nn.ReLU(True),
            #     # nn.Linear(args.hid1, args.hid2),
            #     nn.ReLU(True),
            #     nn.Linear(args.hid1, args.hid2),
            #     nn.ReLU(True)
            # )
            ginnn = nn.Sequential(
                nn.Linear(hid1, hid1),
                # nn.ReLU(True),
                # nn.Linear(args.hid1, args.hid2),
                nn.ReLU(True),
                nn.Linear(hid1, hid1),
                nn.ReLU(True)
            )
            self.gin = DeGINConv(ginnn)
    def forward(self, input,graph, idx=None):
        global adp
        print('input', input.shape)  #[4, 1, 2, 640])
        print('图',graph.shape)#[4, 2, 2])
        # seq_len = input.size(3)  #
        # print('seq_len', seq_len)  # 168 62
        # print('seq_length', self.seq_length)  # 168 62
        # assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

# 全局特征自注意力机制=================================================================================================
        input = torch.squeeze(input,dim=1)
        ds,z = self.attention[0](input)#
        print('自注意力机制', z.shape)  #[4, 3072, 2])
# 图构造=============================================================================================================
        # if self.seq_length<self.receptive_field:
        # input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
        # print('receptive pad',self.receptive_field-self.seq_length)
        # 1.四维Tensor：传入四元素tuple(pad_l, pad_r, pad_t, pad_b)，
        # 指的是（左填充，右填充，上填充，下填充），其数值代表填充次数

        # 图卷积
        # idx = torch.tensor(idx)##
        # if self.gcn_true:
        #     if self.buildA_true:
        #         if idx is None:  # 空
        #             print('adp  none')
        #             adp = self.gc(self.idx)  # 图构建
        #             print('adp none', adp.shape)
        #         else:
        #             print('adp not none')
        #             adp = self.gc(idx)
        #             print('adp not none', adp.shape)  # torch.Size([137, 137])/torch.Size([144, 144])
        #     else:
        #         adp = self.predefined_A
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i, A in enumerate(graph):
            # .cpu()
            A = A / torch.sum(A, 0)
            A = A.reshape(1, A.size(0), A.size(1))
            #  ###每个batch的图
            # self.A_new[i, :, :] = A
            if i == 0:
                A_new = A
            else:
                A_new = torch.cat((A_new, A), dim=0)
        # A_new = self.A_new
        A_new=A_new
# 图卷积=============================================================================================================
        x_conv=z.transpose(2, 1)
        if self.decoder == 'GNN':
            # x0 = F.relu(self.gnn0(x_conv,self.A))
            x1 = F.relu(self.gnn1(x_conv, A_new))  # s
            print('x1', x1.shape)  # [
            x2 = F.relu(self.gnn2(x1, A_new))
            print('x2', x2.shape)  # 1
            x3 = self.gnn3(x2, A_new)
            print('x3', x3.shape)  #

        if self.decoder == 'GCN':
            # x1 = F.relu(self.gcn1(x_conv,self.A))
            x1 = F.relu(self.gcn1(x_conv, A_new))  # self.A
            x2 = F.relu(self.gcn2(x1, A_new))
            x3 = self.gcn3(x2, A_new)
        if self.decoder == 'SAGE':
            x1 = F.relu(self.sage1(x_conv, A_new))
            x2 = F.relu(self.sage2(x1, A_new))
            x3 = F.relu(self.sage3(x2, A_new))
        adjs=A_new
        if self.decoder == 'rGNN':
            x1 = F.relu(self.gc1(x_conv, adjs))
            # x1 = F.dropout(x1, self.dropout)
            x2 = F.relu(self.gc2(x1, adjs))
            x3 = F.relu(self.gc3(x2, adjs))
            # x3 = F.dropout(x2, self.dropout)
            x3 = x3.squeeze(dim=1)
        if self.decoder == 'GIN':
            x3 = F.relu(self.gin(x_conv, A_new))
            x3 = x3.squeeze(dim=1)
        x3 = torch.unsqueeze(x3, dim=1)
        print('图卷积',x3.shape)#[4, 1, 2, 3072])
        x3 = self.start_conv(x3)  # 二维卷积
        #x = self.cbamx(x)
        print('图卷积结束x3',x3.shape)#[4, 16, 2, 3072])
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # z = z.transpose(1, 2)
        # z = torch.unsqueeze(z, dim=1)
        #
        # print('图卷积 z', z.shape)#[16, 1, 6, 3072]
        #
        # x = self.start_conv(z)  # 二维卷积
        # x = self.cbamx(x)
        # print('图卷积 x', x.shape)  # torch.Size([4, 16, 137, 187])/torch.Size([4, 16, 144, 62])
        #
        # print('图卷积开始x', x.shape)  # torch.Size([4, 16, 137, 181])/ torch.Size([4, 16, 144, 62]) [16, 16, 6, 1197])
        # ##图卷积,输入是上一层的x
        # if self.gcn_true:  # 有图
        #     x = self.gconv1[0](x, adp) + self.gconv2[0](x, adp.transpose(1, 0))  # [i] #[i]
        #     print('图卷积第一层', x.shape)  # [16, 32, 6, 3072]

            # poolx, poola = self.Diffpool1(x, adp)
            # print('图池化 第一层', poolx.shape, poola.shape)
            # poolx = self.gconv1[1](poolx, poola) + self.gconv2[1](poolx, poola.transpose(1, 0))
            #
            # poolx, poola = self.Diffpool2(poolx, poola)

            ##关于(x, adp)，(x, adp.transpose(1,0))，+相加操作
        # else:
        #     x = self.residual_convs(x)  # [i]
        #print('图卷积结束 x', x.shape)  # torch.Size([4, 16, 137, 181]), torch.Size([16, 16, 144, 62])#[16, 16, 9, 144]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        x = F.relu(self.end_conv_1(x3))  # 二维卷积
        print('end_conv_1', x.shape)  #
        x = self.end_conv_2(x)  # [4, 16, 2, 3072])
        print('end_conv_2', x.shape)#[4, 64, 2, 3072])
        x = torch.squeeze(x,dim=1)
        print('图NN 结束', x.shape)  #[4, 1, 2, 3072])

        # 串行

        fl1 = x.mean(dim=[1])
        print('mean fl', fl1.shape)#[4, 3072])
        fl = self.mlp_head(fl1)

        print('fl', fl.shape)  # ([4, 3])

        return ds,fl1,fl##用于实验
        #return fl##用于生成cam图


# 二维 CBMA  通道注意力机制和空间注意力机制的结合
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, (1,1), bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, (1,1), bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        return self.sigmoid(out)  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv2d(2, 1, (1,kernel_size), padding=(0,padding), bias=False)##出问题
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        print('空间x',x.shape)#[16, 2, 6, 3072]
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        print('空间x',x.shape)#[16, 1, 12, 3072]
        return self.sigmoid(x)  # sigmoid激活操作


class cbamblock1(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock1, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        print('cbam 通道',x.shape)#[16, 16, 6, 3072]
        #print('self.spatialattention(x)', self.spatialattention(x).shape)#[16, 1, 12, 3072]
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层

        print('cbam 空间', x.shape)
        return x

def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass
