import json
import os
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class CocoKarpathyTest(Dataset):
    """COCO Karpathy-test retrieval images, iterated in manifest order with
    label = row index, so image i pairs with line i of coco_karpathy_test.txt
    (see utils.scripts.prepare_coco_karpathy). Use samples_per_class=None."""

    def __init__(self, root, transform=None):
        with open(os.path.join(root, "coco_karpathy_test_manifest.json")) as f:
            self.records = json.load(f)["records"]
        self.img_dir = os.path.join(root, "val2014")
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.img_dir, rec["filename"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, rec["row"]


def caltech_imagefolder(root, transform=None):
    """ImageFolder loader that drops the Caltech-101 'BACKGROUND_Google' pseudo-class and remaps the
    remaining labels to a contiguous 0..N-1 range, so they match caltech_101_classes (index == label).

    torchvision's ImageFolder otherwise sees 102 folders with 'BACKGROUND_Google' sorting to index 0,
    which shifts every real class by +1 and misaligns them with the class-name list / classifier.
    A no-op for any dataset that has no 'BACKGROUND_Google' folder, so it is safe as a drop-in for the
    generic ImageFolder path. Works whether or not BACKGROUND_Google is present on disk, keeping the
    idx->class map and the extracted activations consistent."""
    ds = ImageFolder(root=root, transform=transform)
    bg = ds.class_to_idx.get("BACKGROUND_Google")
    if bg is None:
        return ds
    ds.samples = [(p, l - 1 if l > bg else l) for p, l in ds.samples if l != bg]
    ds.imgs = ds.samples
    ds.targets = [l for _, l in ds.samples]
    ds.classes = [c for c in ds.classes if c != "BACKGROUND_Google"]
    ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}
    return ds


def dataset_to_dataloader(dataset, samples_per_class = 5, tot_samples_per_class=50, batch_size=8, shuffle=False, num_workers=8, seed=0):
    if samples_per_class is None:
        print("Full dataset")
        dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)        
    else:
        # Take a subset
        random.seed(seed)
        all_indices = list(range(len(dataset)))
        nr_classes = len(dataset.classes)
        index = [random.sample(all_indices[x*tot_samples_per_class:(x + 1)*tot_samples_per_class], samples_per_class) for x in range(nr_classes)]
        index = [x for xs in index for x in xs]
        dataset = torch.utils.data.Subset(dataset, index)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    return dataloader

def dataset_subset(dataset, samples_per_class = 5, tot_samples_per_class=50, seed=42):
    if samples_per_class is None:
        print("Full dataset")
    else:
        # Take a subset
        random.seed(seed)
        all_indices = list(range(len(dataset)))
        nr_classes = len(dataset.classes)
        index = [random.sample(all_indices[x*tot_samples_per_class:(x + 1)*tot_samples_per_class], samples_per_class) for x in range(nr_classes)]
        index = [x for xs in index for x in xs]
        dataset = torch.utils.data.Subset(dataset, index)


    return dataset



    