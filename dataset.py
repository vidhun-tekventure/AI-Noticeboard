from torch.utils.data import Dataset
from PIL import Image
import json
import os
import torch
from torchvision import transforms

class NoticeboardDataset(Dataset):
    def __init__(self, image_dir, label_json):
        with open(label_json, "r", encoding='utf-8') as f:
            self.labels = json.load(f)
        self.image_dir = image_dir
        
        # Define the criteria explicitly (skip "image", "rules", "overall_compliance")
        self.criteria = [
            "has_arabic", 
            "has_english", 
            "approved_name", 
            "design_compliant", 
            "no_obstruction", 
            "well_lit"
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.labels[idx]
        image_path = os.path.join(self.image_dir, entry["image"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Cannot open image {image_path}: {e}")

        image = self.transform(image)
        
        # Convert boolean labels to float (0.0 or 1.0)
        label = torch.tensor([
            float(entry[c]) if entry[c] is not None else 0.0  # Handle None values
            for c in self.criteria
        ])
        
        return image, label