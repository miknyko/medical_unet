
import torch
import numpy as np
from pydicom import dcmread


class DicomDataset(torch.utils.data.Dataset):
    def __init__(self, images_list, masks):
        self.images_list = images_list
        self.masks = masks

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        image = dcmread(image_path)
        image = image.pixel_array
        normalized = image.astype(np.float32)

        # load the mask from memory
        key = image_path.parent.stem + "_seg"  # e.g. Case_000_seg
        index = (
            int(image_path.stem.split("-")[1].strip("0")) - 1
        )  # e.g. 1-028.dcm -> 27

        mask = self.masks[key][index]

        # convert to tensor
        image = torch.from_numpy(normalized).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, image_path.parent.stem + "+" + image_path.stem


def main():

    # masks
    mask_name = "Dataset/Segmentations/Case_000_seg.npz"
    data = np.load(mask_name)["masks"]
    print(data.shape)
    print(data)
    print(np.max(data))


if __name__ == "__main__":
    main()
