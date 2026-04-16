import segmentation_models_pytorch as smp


class DeepLabV3Plus:
    @staticmethod
    def build(num_classes=21, encoder_name="resnet50", encoder_weights="imagenet",
              output_stride=16):
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            encoder_output_stride=output_stride,
        )
        return model
