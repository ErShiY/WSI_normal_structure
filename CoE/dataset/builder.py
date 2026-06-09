import glob
import h5py
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

slide_label_mapping = {
    "negative": 0,
    "itc": 1,
    "micro": 2,
    "macro": 3
}

patient_label_mapping = {
    "pN0": 0,
    "pN0(i+)": 1,
    "pN1mi": 2,
    "pN1": 3,
    "pN2": 4
}

label_mapping_nsclc = {
    "LUAD": 0,
    "LUSC": 1
}

label_mapping_brca = {
    "IDC": 0,
    "ILC": 1
}


def convert_slide_id(slide_id):
    """'0010' -> 'patient_001_node_0'"""
    slide_id = f"{int(slide_id):04d}"
    patient_str = slide_id[:3]
    node_str = slide_id[3:]
    return f"patient_{patient_str}_node_{node_str}"


class CSVReader:
    def __init__(self, cfg):
        self.csv_dir = cfg["path"]["csv"]
        self.fold = cfg["fold"]
        self.target_dataset = cfg["dataset"]

        self.csv_files = {
            "train": os.path.join(self.csv_dir, f"train_{self.fold}.csv"),
            "val": os.path.join(self.csv_dir, f"val_{self.fold}.csv"),
            "test": os.path.join(self.csv_dir, f"test_{self.fold}.csv")
        }

    def __getitem__(self, split):
        if split not in self.csv_files:
            raise ValueError(
                f"Invalid split name: {split}. Must be one of {list(self.csv_files.keys())}."
            )

        data_list = []

        if self.target_dataset == "camelyon17":
            df = pd.read_csv(
                self.csv_files[split],
                dtype={"patient_id_num": str}
            )

            slides = df["patient_id_num"].tolist()
            slide_labels = df["slide_stage"].tolist()
            patient_labels = df["patient_stage"].tolist()

            for i, slide in enumerate(slides):
                item = {
                    "slide": slide,
                    "slide_label": slide_labels[i],
                    "patient_label": patient_labels[i]
                }
                data_list.append(item)

        elif self.target_dataset in ["tcga-brca", "tcga-nsclc"]:
            df = pd.read_csv(self.csv_files[split])

            slides = df["patient_id"].tolist()
            slide_labels = df["subtype"].tolist()

            for i, slide in enumerate(slides):
                item = {
                    "slide": slide,
                    "slide_label": slide_labels[i]
                }
                data_list.append(item)

        else:
            raise ValueError(f"Unsupported dataset: {self.target_dataset}")

        return data_list


class WSIDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.data_list = CSVReader(cfg)[split]

        self.target_dataset = cfg["dataset"]
        self.h5_path = cfg["path"].get("h5", None)
        self.pt_path = cfg["path"].get("pt", None)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]

        if self.target_dataset == "camelyon17":
            slide_id = entry["slide"]
            slide_name = convert_slide_id(slide_id)

            h5_file = os.path.join(self.h5_path, f"{slide_name}.h5")

            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"No h5 file found: {h5_file}")

            with h5py.File(h5_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32)
                coords = torch.tensor(f["coords"][:], dtype=torch.int32)

            if features.dim() == 2:
                features = features.unsqueeze(0)

            sample = {
                "slide": slide_name,
                "features": features,
                "coords": coords,
                "slide_label": slide_label_mapping[entry["slide_label"]],
                "patient_label": patient_label_mapping[entry["patient_label"]]
            }

        elif self.target_dataset in ["tcga-brca", "tcga-nsclc"]:
            slide_id = entry["slide"]
            pt_files = glob.glob(os.path.join(self.pt_path, f"{slide_id}*.pt"))

            if len(pt_files) == 0:
                raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")

            features = torch.load(pt_files[0], map_location="cpu")

            label_mapping = (
                label_mapping_nsclc
                if self.target_dataset == "tcga-nsclc"
                else label_mapping_brca
            )

            sample = {
                "slide": slide_id,
                "features": features,
                "slide_label": label_mapping[entry["slide_label"]]
            }

        else:
            raise ValueError(f"Unsupported dataset: {self.target_dataset}")

        return sample


def build_dataloader(cfg):
    train_dataset = WSIDataset(cfg, "train")
    val_dataset = WSIDataset(cfg, "val")
    test_dataset = WSIDataset(cfg, "test")

    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
