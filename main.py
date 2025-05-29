from fastapi import FastAPI, UploadFile, File
from model import NoticeboardModel
from torchvision import transforms
from PIL import Image
import torch
from typing import Dict

app = FastAPI()

criteria = ["has_arabic", "has_english", "approved_name",
            "design_compliant", "no_obstruction", "well_lit"]

rules = {
    "has_arabic": {
        "requirement": "Arabic Name Display: The shop name must be prominently displayed in Arabic.",
        "severity": "high"
    },
    "has_english": {
        "requirement": "English translation must be smaller than the Arabic text.",
        "severity": "medium"
    },
    "approved_name": {
        "requirement": "Only the name approved by the Licensing Committee should be used.",
        "severity": "high"
    },
    "design_compliant": {
        "requirement": "The signboard must adhere to technical conditions for uniformity.",
        "severity": "medium"
    },
    "no_obstruction": {
        "requirement": "The signboard should not obstruct architectural elements or windows.",
        "severity": "low"
    },
    "well_lit": {
        "requirement": "Signboard must be well-lit, especially for 24/7 shops.",
        "severity": "medium"
    }
}

# Load image classification model
model = NoticeboardModel(num_labels=len(criteria))
model.load_state_dict(torch.load("notice_model.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def determine_compliance_status(validation_results: Dict[str, bool]) -> str:
    passed = sum(validation_results.values())
    total = len(validation_results)

    if passed == total:
        return "Fully Compliant"
    elif passed >= total * 0.7:
        return "Mostly Compliant"
    elif passed >= total * 0.3:
        return "Partially Compliant"
    else:
        return "Non-Compliant"

def generate_issues_list(validation_results: Dict[str, bool]) -> list:
    issues = []
    for criterion, passed in validation_results.items():
        if not passed:
            issues.append({
                "criterion": criterion,
                "requirement": rules[criterion]["requirement"],
                "severity": rules[criterion]["severity"],
                "message": f"Failed {criterion.replace('_', ' ')} requirement"
            })
    return issues

@app.post("/validate")
async def validate_noticeboard(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0].numpy()

    validation_results = {criteria[i]: bool(output[i] > 0.5) for i in range(len(criteria))}
    compliance_status = determine_compliance_status(validation_results)
    issues = generate_issues_list(validation_results)

    return {
        "compliance_status": compliance_status,
        "validation_details": {
            "passed_checks": sum(validation_results.values()),
            "total_checks": len(validation_results),
            "results": validation_results,
            "critical_issues": [issue for issue in issues if issue['severity'] == 'high'],
            "other_issues": [issue for issue in issues if issue['severity'] != 'high']
        },
        "image_metadata": {
            "size": img.size,
            "mode": img.mode
        }
    }

@app.get("/")
async def health_check():
    return {"status": "active", "model": "noticeboard_validator"}
