
from train import main
from train import get_args_parser





MODELS = [
("resnet18", "ResNet18_Weights.IMAGENET1K_V1"),
("alexnet", "AlexNet_Weights.IMAGENET1K_V1"),
("convnext_base", "ConvNeXt_Base_Weights.IMAGENET1K_V1"),
("convnext_large", "ConvNeXt_Large_Weights.IMAGENET1K_V1"),
("convnext_small", "ConvNeXt_Small_Weights.IMAGENET1K_V1"),
("convnext_tiny", "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"),
("densenet121", "DenseNet121_Weights.IMAGENET1K_V1"),
("densenet161", "DenseNet161_Weights.IMAGENET1K_V1"),
("densenet169", "DenseNet169_Weights.IMAGENET1K_V1"),
("densenet201", "DenseNet201_Weights.IMAGENET1K_V1"),
("efficientnet_b0", "EfficientNet_B0_Weights.IMAGENET1K_V1"),
("efficientnet_b1", "EfficientNet_B1_Weights.IMAGENET1K_V1"),
("efficientnet_b2", "EfficientNet_B2_Weights.IMAGENET1K_V1"),
("efficientnet_b3", "EfficientNet_B3_Weights.IMAGENET1K_V1"),
("efficientnet_b4", "EfficientNet_B4_Weights.IMAGENET1K_V1"),
("efficientnet_b5", "EfficientNet_B5_Weights.IMAGENET1K_V1"),# DONE
("efficientnet_b6", "EfficientNet_B6_Weights.IMAGENET1K_V1"),
("efficientnet_b7", "EfficientNet_B7_Weights.IMAGENET1K_V1"),
("efficientnet_v2_l", "EfficientNet_V2_L_Weights.IMAGENET1K_V1"),
("efficientnet_v2_m", "EfficientNet_V2_M_Weights.IMAGENET1K_V1"),
("efficientnet_v2_s", "EfficientNet_V2_S_Weights.IMAGENET1K_V1"),
("googlenet", "GoogLeNet_Weights.IMAGENET1K_V1"),
("inception_v3", "Inception_V3_Weights.IMAGENET1K_V1"),
("mnasnet0_5", "MNASNet0_5_Weights.IMAGENET1K_V1"),
("mnasnet0_75", "MNASNet0_75_Weights.IMAGENET1K_V1"),
("mnasnet1_0", "MNASNet1_0_Weights.IMAGENET1K_V1"),
("mnasnet1_3", "MNASNet1_3_Weights.IMAGENET1K_V1"),
("maxvit_t", "MaxVit_T_Weights.IMAGENET1K_V1"),
("mobilenet_v2", "MobileNet_V2_Weights.IMAGENET1K_V1"),
("mobilenet_v3_large", "MobileNet_V3_Large_Weights.IMAGENET1K_V1"),
("mobilenet_v3_small", "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
("regnet_x_16gf", "RegNet_X_16GF_Weights.IMAGENET1K_V1"),
("regnet_x_1_6gf", "RegNet_X_1_6GF_Weights.IMAGENET1K_V1"),
("regnet_x_32gf", "RegNet_X_32GF_Weights.IMAGENET1K_V1"),
("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights.IMAGENET1K_V1"),
("regnet_x_400mf", "RegNet_X_400MF_Weights.IMAGENET1K_V1"),
("regnet_x_800mf", "RegNet_X_800MF_Weights.IMAGENET1K_V1"),
("regnet_x_8gf", "RegNet_X_8GF_Weights.IMAGENET1K_V1"),
("regnet_y_128gf", "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1"),
("regnet_y_128gf", "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),
("regnet_y_16gf", "RegNet_Y_16GF_Weights.IMAGENET1K_V1"),
("regnet_y_16gf", "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1"),
("regnet_y_16gf", "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),
("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights.IMAGENET1K_V1"),
("regnet_y_32gf", "RegNet_Y_32GF_Weights.IMAGENET1K_V1"),
("regnet_y_32gf", "RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1"),
("regnet_y_32gf", "RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),
("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1"),
("regnet_y_400mf", "RegNet_Y_400MF_Weights.IMAGENET1K_V1"),
("regnet_y_800mf", "RegNet_Y_800MF_Weights.IMAGENET1K_V1"),
("regnet_y_8gf", "RegNet_Y_8GF_Weights.IMAGENET1K_V1"),
("resnext101_32x8d", "ResNeXt101_32X8D_Weights.IMAGENET1K_V1"),
("resnext101_64x4d", "ResNeXt101_64X4D_Weights.IMAGENET1K_V1"),
("resnext50_32x4d", "ResNeXt50_32X4D_Weights.IMAGENET1K_V1"),
("resnet101", "ResNet101_Weights.IMAGENET1K_V1"),
("resnet152", "ResNet152_Weights.IMAGENET1K_V1"),
("resnet18", "ResNet18_Weights.IMAGENET1K_V1"),
("resnet34", "ResNet34_Weights.IMAGENET1K_V1"),
("resnet50", "ResNet50_Weights.IMAGENET1K_V1"),
("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),
("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),
("shufflenet_v2_x1_5", "ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1"),
("shufflenet_v2_x2_0", "ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1"),
("squeezenet1_0", "SqueezeNet1_0_Weights.IMAGENET1K_V1"),
("squeezenet1_1", "SqueezeNet1_1_Weights.IMAGENET1K_V1"),
("swin_b", "Swin_B_Weights.IMAGENET1K_V1"),
("swin_s", "Swin_S_Weights.IMAGENET1K_V1"),
("swin_t", "Swin_T_Weights.IMAGENET1K_V1"),
("swin_v2_b", "Swin_V2_B_Weights.IMAGENET1K_V1"),
("swin_v2_s", "Swin_V2_S_Weights.IMAGENET1K_V1"),
("swin_v2_t", "Swin_V2_T_Weights.IMAGENET1K_V1"),
("vgg11_bn", "VGG11_BN_Weights.IMAGENET1K_V1"),
("vgg11", "VGG11_Weights.IMAGENET1K_V1"),
("vgg13_bn", "VGG13_BN_Weights.IMAGENET1K_V1"),
("vgg13", "VGG13_Weights.IMAGENET1K_V1"),
("vgg16_bn", "VGG16_BN_Weights.IMAGENET1K_V1"),
("vgg16", "VGG16_Weights.IMAGENET1K_V1"),
("vgg16", "VGG16_Weights.IMAGENET1K_FEATURES"),
("vgg19_bn", "VGG19_BN_Weights.IMAGENET1K_V1"),
("vgg19", "VGG19_Weights.IMAGENET1K_V1"),
("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_V1"),
("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1"),
("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1"),
("vit_b_32", "ViT_B_32_Weights.IMAGENET1K_V1"),
("vit_h_14", "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1"),
("vit_h_14", "ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1"),
("vit_l_16", "ViT_L_16_Weights.IMAGENET1K_V1"),
("vit_l_16", "ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1"),
("vit_l_16", "ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1"),
("vit_l_32", "ViT_L_32_Weights.IMAGENET1K_V1"),
("wide_resnet101_2", "Wide_ResNet101_2_Weights.IMAGENET1K_V1"),
("wide_resnet50_2", "Wide_ResNet50_2_Weights.IMAGENET1K_V1"),]

MODELS = MODELS[:1]
results = []
for model, weight in MODELS:
    for use_nncf in [False, True]:
        nncf_flag = ["--nncf"] if use_nncf else []
        args = get_args_parser().parse_args(f"--data-path /home/dlyakhov/datasets/imagenet/ --model {model} --test-only --weights {weight} --batch 64 ".split() + nncf_flag)
        exc = ""
        try:
            result = main(args)
        except:
            from traceback import format_exc
            exc = format_exc()
            print(exc)
            result = -1
        results.append((model, weight, use_nncf, result, exc))


print(results)
with open("result.txt", "w") as f:
    f.write("\n".join(list(map(str, results))))