import os

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights


class ImageData:
    def __init__(self, folder: str):
        self.D = folder

    def load_images(self) -> list:
        """
        Load images
        :return: the images list
        """
        inner_images = []
        for f in os.listdir(self.D):
            if f.endswith(".jpg") or f.endswith(".png"):
                inner_images.append(Image.open(os.path.join(self.D, f)))
        return inner_images


class ImgProcess:
    def __init__(self, size: int):
        self.s = size

    def resize_and_gray(self, img_list: list) -> list:
        """
        Resize and gray the image
        :param img_list: the images list
        :return: the list of transformed images
        """
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

    def predict_img(self, inner_images: list) -> list[int | float | bool]:
        """
        Predict a list of images
        :param inner_images:  the list of images
        :return: the list of predictions
        """
        tmp_results = []
        for img_tensor in inner_images:
            inner_pred = self.mdl(img_tensor.unsqueeze(0))
            tmp_results.append(torch.argmax(inner_pred, dim=1).item())
        return tmp_results


if __name__ == "__main__":
    writer = SummaryWriter("tensorboard/runs/image_classification")

    loader = ImageData("../images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    preprocessed_tensor = processor.resize_and_gray(images)

    predictor = Predictor()
    results = predictor.predict_img(preprocessed_tensor)

    for i, tensor in enumerate(preprocessed_tensor):
        writer.add_image(f"{results[i]}", tensor, 0)

    writer.close()

    print(f"Predicted {len(results)} images: {results}")
