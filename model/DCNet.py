import torch
import torch.nn as nn
from torch.nn import Softmax
import torchvision.models as models
from lib.pvt import pvt_v2_b2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv2d, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

class SPPFCSPC(nn.Module):
    
    def __init__(self, in_channels, out_channels=32, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        hid_channels = int(2 * out_channels * e)  # hidden channels
        self.cv1 = conv(in_channels, hid_channels, 1, 1)
        self.cv2 = conv(in_channels, hid_channels, 1, 1)
        self.cv3 = conv(hid_channels, hid_channels, 3, 1)
        self.cv4 = conv(hid_channels, hid_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = conv(4 * hid_channels, hid_channels, 1, 1)
        self.cv6 = conv(hid_channels, hid_channels, 3, 1)
        self.cv7 = conv(2 * hid_channels, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
    
######################dynamic conv############################################
class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K,temprature=30,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=self.net(att).view(x.shape[0],-1) #bs,K
        return F.softmax(att/self.temprature,-1)

class DynamicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0,dilation=1,grounps=1,bias=True,K=4,temprature=30,ratio=4,init_weight=True):
        super().__init__()
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=grounps
        self.bias=bias
        self.K=K
        self.init_weight=init_weight
        self.attention=Attention(in_planes=in_planes,ratio=ratio,K=K,temprature=temprature,init_weight=init_weight)

        self.weight=nn.Parameter(torch.randn(K,out_planes,in_planes//grounps,kernel_size,kernel_size),requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(K,out_planes),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs,in_planels,h,w=x.shape
        softmax_att=self.attention(x) #bs,K
        x=x.view(1,-1,h,w)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.out_planes,self.in_planes//self.groups,self.kernel_size,self.kernel_size) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        
        output=output.view(bs,self.out_planes,h,w)
        return output

######################dynamic conv############################################
# Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



def INF(B, H, W, device):
    # 修改INF函数，确保返回的张量在正确的设备上
    return -torch.diag(torch.tensor(float("inf")).repeat(H).to(device), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device  # 获取输入张量所在的设备
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # 修改INF函数，确保它返回在相同设备上的张量
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + INF(m_batchsize, height, width, device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(in_channel, reduction_ratio)
        self.SpatialGate = SpatialAttention()

    def forward(self, x):
        channel_att = self.ChannelGate(x)
        x = channel_att * x
        spatial_att = self.SpatialGate(x)
        x = spatial_att * x
        return x

# Cross-modal Fusion
class CFF(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(CFF, self).__init__()
        # self.cross = CrossAttentionModule(in_channel)
        self.fea_fus = CBAM(in_channel)

    def forward(self, img, gn):
        x = img + gn + (img * gn)
        x = self.fea_fus(x)
        # print("x shape:", x.shape)
        return x

class iAFF(nn.Module):

	"""
	implimenting iAFF module
	"""

	def __init__(self, channels, r=4):
		super(iAFF, self).__init__()
		inter_channels = int(channels // r)

		self.local_attention1 = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
		self.global_attention1 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)

		self.local_attention2 = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
		self.global_attention2 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)

		self.sigmoid = nn.Sigmoid()


	def forward(self, in_feat , iaff_feat):
		"""
		Implimenting the iAFF forward step
		"""
		x = in_feat
		y = iaff_feat
		xa = x+y
		xl = self.local_attention1(xa)
		xg = self.global_attention1(xa)
		xlg = xl+xg
		m1 = self.sigmoid(xlg)
		xuniony = x * m1 + y * (1-m1)

		xl2 = self.local_attention2(xuniony)
		xg2 = self.global_attention2(xuniony)
		xlg2 = xl2 + xg2
		m2 = self.sigmoid(xlg2)
		z = x * m2 + y * (1-m2)
		return z
    
class LA(nn.Module):
	def __init__(self, channels, r=16):
		super(LA, self).__init__()
		inter_channels = int(channels // r)

		self.local_attention = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
			nn.Sigmoid(),
		)
		self.global_attention = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
			nn.Sigmoid(),
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		"""
		Implimenting the iAFF forward step
		"""
		x_la = self.local_attention(x)
		x_ga = self.global_attention(x)
		res = self.sigmoid(x_la + x_ga)
		out = res * x
		return out



class GNG(nn.Module):
    def __init__(self, in_channels, kernel_size=1, bias=False):
        super(GNG, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.Sigmoid()
        )
        self.out_cov = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, in_feat):
        attention_map = self.gate_conv(in_feat)
        # in_feat = (in_feat * (attention_map + 1))
        out_feat = self.out_cov(attention_map)
        return out_feat


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class GAR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(GAR, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_cov = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, in_feat, gate_feat):
        attention_map = self.gate_conv(torch.cat([in_feat, gate_feat], dim=1))
        in_feat = (in_feat * (attention_map + 1))
        out_feat = self.out_cov(in_feat)
        return out_feat


class LVB(nn.Module):
    def __init__(self, in_channels, reduction, kernel_size, bias, act):
        super(LVB, self).__init__()
        modules_body = []
        modules_body.append(conv(in_channels, in_channels, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(in_channels, in_channels, kernel_size, bias=bias))

        self.CA = CALayer(in_channels, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.la = LA(in_channels , r = 16)
        self.cc_attention = CrissCrossAttention(in_channels)
        self.outconv = BasicConv2d(2 * in_channels, in_channels, 3) ####1->3
        self.sigmoid = nn.Sigmoid()
    # def forward(self, x):
    #     xla = self.la(x)
    #     xga = self.cc_attention(x)
    #     cat = torch.cat((xla, xga), dim=1)
    #     res = self.outconv(cat)
    #     res = self.CA(res)
    #     out = res + x

    #     return out
    def forward(self, x):
        xc = self.body(x)
        xla = self.la(xc)
        xga = self.cc_attention(xc)
        cat = torch.cat((xla, xga), dim=1)
        res = self.outconv(cat)
        res = self.CA(res)
        out = res + x

        return out   

class LVD(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias, act, n_resblocks):
        super(LVD, self).__init__()
        modules_body = [LVB(channel, kernel_size, reduction, bias=bias, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



class DCNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3, reduction=4,embed_dims=[64, 128, 320, 512], bias=False, act=nn.PReLU(), n_resblocks=2, iteration=2):
        super(DCNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.iteration = iteration

        self.dc_4 = DynamicConv(embed_dims[0], 32, 3, 1, 1)
        self.dc_3 = DynamicConv(embed_dims[1], 32, 3, 1, 1)
        self.dc_2 = DynamicConv(embed_dims[2], 32, 3, 1, 1)
        self.dc_1 = DynamicConv(embed_dims[3], 32, 3, 1, 1)

        self.gng4 = GNG(channel)
        self.gng3 = GNG(channel)
        self.gng2 = GNG(channel)
        self.gng1 = GNG(channel)

        self.fusion_4 = CFF(channel)
        self.fusion_3 = CFF(channel)
        self.fusion_2 = CFF(channel)
        self.fusion_1 = CFF(channel)

        self.iaff = iAFF(channel, r = 4)

        self.lvd_1 = LVD(channel, kernel_size, reduction, bias, act, n_resblocks)  # 32 x 11 x 11
        self.lvd_2 = LVD(2 * channel, kernel_size, reduction, bias, act, n_resblocks)  # 64 x 22 x 22
        self.lvd_3 = LVD(3 * channel, kernel_size, reduction, bias, act, n_resblocks)  # 96 x 44 x 44

        self.convlayer = BasicConv2d(channel, channel, kernel_size=1, padding=0)
        self.gate_conv_1 = BasicConv2d(channel, 1, 1)
        self.gate_conv_2 = BasicConv2d(2*channel, 1, 1)
        self.gate_conv = nn.Sequential(
            BasicConv2d(channel, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        )

        self.unsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.pred = nn.Conv2d(channel, 1, 1)

        self.Fus = SPPFCSPC(2 * channel)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_pred = nn.Conv2d(channel, 1, 1)

    def forward(self, x):  # x=img+gn

        pvt = self.backbone(x)
        x4 = pvt[0]  # 64x88x88
        x3 = pvt[1]  # 128x44x44
        x2 = pvt[2]  # 320x22x22
        x1 = pvt[3]  # 512x11x11

        x4 = self.dc_4(x4)
        x3 = self.dc_3(x3)
        x2 = self.dc_2(x2)
        x1 = self.dc_1(x1)

        # x4_img= self.convlayer(x4)
        # x4_gn = self.gng4(x4)
        # x3_img= self.convlayer(x3)
        # x3_gn = self.gng3(x3)
        # x2_img= self.convlayer(x2)
        # x2_gn = self.gng2(x2)
        # x1_img= self.convlayer(x1)
        # x1_gn = self.gng1(x1)
        x4_img= x4
        x4_gn = self.gng4(x4)
        x3_img= x3
        x3_gn = self.gng3(x3)
        x2_img= x2
        x2_gn = self.gng2(x2)
        x1_img= x1
        x1_gn = self.gng1(x1)


        stage_pred = list()
        rough_pred = None  # stage pre
        for iter in range(self.iteration):
            x1 = self.fusion_1(x1_img, x1_gn)
            if rough_pred == None:
                x1 = x1
            else:
                rough_pred = self.gate_conv(rough_pred)
                # print(rough_pred.shape)
                # print(x1.shape)
                x1 = self.iaff(x1, rough_pred)
            x2_feed = self.lvd_1(x1)
            x2 = self.fusion_2(x2_img, x2_gn)
            if iter > 0:
                x2_gate = self.unsample_2(self.gate_conv_1(x2_feed))
                x2 = self.iaff(x2, x2_gate)
            x3_feed = self.lvd_2(torch.cat((x2, self.unsample_2(x2_feed)), dim=1))
            x3 = self.fusion_3(x3_img, x3_gn)
            if iter > 0:
                x3_gate = self.unsample_2(self.gate_conv_2(x3_feed))
                x3 = self.iaff(x3, x3_gate)
            x4_feed = self.lvd_3(torch.cat((x3, self.unsample_2(x3_feed)), dim=1))
            rough_pred = self.out(x4_feed)
            out_map = self.pred(rough_pred)
            pred = F.interpolate(out_map, scale_factor=8, mode='bilinear')
            stage_pred.append(pred)

        x4 = self.fusion_4(x4_img, x4_gn)
        x4_out = self.downsample(x4)
        x_in = torch.cat((rough_pred, x4_out), dim=1)
        # refined_pred = self.SAM(rough_pred, x4_out)
        refined_pred = self.Fus(x_in)

        pred2 = self.out_pred(refined_pred)
        final_pred = F.interpolate(pred2, scale_factor=8, mode='bilinear')
        return stage_pred, final_pred


if __name__ == '__main__':
    model = DCNet().cuda()
    input_tensor = torch.randn(10, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1[0].size(), prediction2.size())










