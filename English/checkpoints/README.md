# English FastSpeech 2 Checkpoints

Due to GitHub's 100 MB per-file tracking limit, binary PyTorch checkpoint weights (`.pt` files, ~128 MB each) are stored on Google Drive and excluded from Git commits via `.gitignore`.

### Checkpoint Files Available on Google Drive

| Checkpoint Name | File Size | Training Steps | Validation Total Loss | Base Mel Loss | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| `step_8000.pt` | ~128 MB | 8,000 | 2.8872 | 0.854 | 115-speaker base Svarah model |
| `best.pt` | ~128 MB | 7,000 | **2.8830** | 0.852 | Best validation loss checkpoint (115 speakers) |
| `ft_top3_clean_best.pt` | ~128 MB | 2,500 (ft) | **0.9930** | 0.460 | Targeted acoustic adaptation on clean speakers |
| `ft_top4_clean_6000steps.pt`| ~128 MB | 6,000 (ft) | **0.9520** | 0.435 | Deep fine-tuning on top-4 clean voices |

### Google Drive Storage Path
- Primary: `/content/drive/MyDrive/project/English/checkpoints/`
- Mirror: `/content/drive/MyDrive/project (1)/English/checkpoints/`

### Loading Checkpoint for Inference
```python
import torch
from model import FastSpeech2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("English/checkpoints/best.pt", map_location=device)
model = FastSpeech2(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
```
