import os

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights


class ImageData:
    def __init__(self, DIR: str):
        self.D = DIR

    def LoadImages(self) -> list:
        inner_images = []
        for F in os.listdir(self.D):
            if F.endswith(".jpg") or F.endswith(".png"):
                inner_images.append(Image.open(os.path.join(self.D, F)))
        return inner_images


class ImgProcess:
    def __init__(self, size: int):
        self.s = size

    def resize_and_GRAY(self, img_list: list) -> list:
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.s, self.s)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            p_images.append(t(img))
        return p_images


class Predictor:
    def __init__(self):
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def Predict_Img(self, inner_images: list) -> list[int | float | bool]:
        tmp_results = []
        for img_tensor in inner_images:
            inner_pred = self.mdl(img_tensor.unsqueeze(0))
            tmp_results.append(torch.argmax(inner_pred, dim=1).item())
        return tmp_results


if __name__ == "__main__":
    writer = SummaryWriter("tensorboard/runs/image_classification")

    loader = ImageData("../images/")
    images = loader.LoadImages()

    processor = ImgProcess(256)
    preprocessed_tensor = processor.resize_and_GRAY(images)

    predictor = Predictor()
    results = predictor.Predict_Img(preprocessed_tensor)

    for i, tensor in enumerate(preprocessed_tensor):
        writer.add_image(f"{results[i]}", tensor, 0)

    writer.close()

    print(f"Predicted {len(results)} images: {results}")
