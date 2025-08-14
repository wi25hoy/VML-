# -*- coding: utf-8 -*-
import imagenet_models
import mnist_models_clf
import resnet_cifar
import shufflenetv2_clf
import vgg_clf

linearnet = resnet_cifar.linearnet
hiddennet1 = resnet_cifar.hiddennet1
hiddennet2 = resnet_cifar.hiddennet2
resnet6 = resnet_cifar.resnet6
resnet10 = resnet_cifar.resnet10
resnet14 = resnet_cifar.resnet14
resnet14_0125 = resnet_cifar.resnet14_0125
resnet14_025 = resnet_cifar.resnet14_025
resnet14_050 = resnet_cifar.resnet14_050
resnet14_2 = resnet_cifar.resnet14_2
resnet14_4 = resnet_cifar.resnet14_4
resnet14_8 = resnet_cifar.resnet14_8
resnet18 = resnet_cifar.resnet18
resnet20 = resnet_cifar.resnet20
resnet32 = resnet_cifar.resnet32
resnet44 = resnet_cifar.resnet44
resnet56 = resnet_cifar.resnet56
resnet110 = resnet_cifar.resnet110
resnet1202 = resnet_cifar.resnet1202
vgg11 = vgg_clf.vgg11
vgg13 = vgg_clf.vgg13
vgg16 = vgg_clf.vgg16
shufflenet05 = shufflenetv2_clf.shufflenet05
shufflenet1 = shufflenetv2_clf.shufflenet1

mnisthiddennet1 = mnist_models_clf.mnisthiddennet1
mnisthiddennet2 = mnist_models_clf.mnisthiddennet2
mnistlinearnet = mnist_models_clf.mnistlinearnet
mnistnet = mnist_models_clf.mnistnet

imagenet_resnet18 = imagenet_models.resnet18
imagenet_resnet34 = imagenet_models.resnet34
imagenet_resnet50 = imagenet_models.resnet50

AdaptiveLabelVariancePenalty = resnet_cifar.AdaptiveLabelVariancePenalty
