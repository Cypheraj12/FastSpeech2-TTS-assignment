# Telugu FastSpeech 2 Checkpoints

Due to GitHub's 100 MB per-file tracking limit, binary PyTorch checkpoint weights (`.pt` files, ~128 MB each) are stored on Google Drive and excluded from Git commits via `.gitignore`.

### Checkpoint Files Available on Google Drive

| Checkpoint Name | File Size | Training Steps | Validation Total Loss | Base Mel Loss | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| `step_2000.pt` | ~128 MB | 2,000 | 2.5120 | 0.824 | Initial convergence checkpoint |
| `step_4000.pt` | ~128 MB | 4,000 | 2.2450 | 0.710 | Mid-training checkpoint |
| `step_6000.pt` | ~128 MB | 6,000 | **2.1991** | 0.685 | Best validation loss checkpoint |
| `best.pt` | ~128 MB | 6,000 | **2.1991** | 0.685 | Official submission checkpoint |

### Download Link
- [Google Drive Telugu Checkpoints Folder](https://drive.google.com/drive/folders/1754k2h3_iwyKGWcj_v0_TZbN0WACzoIt?usp=sharing)

### Google Drive Storage Path
- Primary: `/content/drive/MyDrive/project/Telugu/checkpoints/`
- Mirror: `/content/drive/MyDrive/project (1)/Telugu/checkpoints/`

### Loading Checkpoint for Inference
```python
import torch
from model import FastSpeech2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("Telugu/checkpoints/best.pt", map_location=device)
model = FastSpeech2(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
```
