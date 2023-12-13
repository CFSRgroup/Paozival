


from feature_map_backbone.aff_net.aff_resnet import ResNet,BasicBlock


class MyAFF_ResNet(ResNet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,xb):
        x=self.maxpool(self.relu(self.bn1(self.conv1(xb))))

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        return x
    
