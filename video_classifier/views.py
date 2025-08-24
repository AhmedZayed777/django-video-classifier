import torch
from torchvision.models.video import r3d_18
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import torchvision.transforms as transforms

# 1️⃣ Load model with correct number of classes
num_classes = 2  # <-- adjust to your trained model
model = r3d_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load checkpoint
checkpoint = torch.load("r3d18_model.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# 2️⃣ Helper function: Convert video to frames
def video_to_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    cap.release()
    return frames

# -----------------------------
# 3️⃣ Preprocess frames
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989]),
])

# Replace your old preprocess_frames with this
def preprocess_frames(frames):
    # Convert list of frames [T, H, W, C] -> tensor
    frames_tensor = torch.stack([transform(f) for f in frames])  # [T, C, H, W]
    frames_tensor = frames_tensor.unsqueeze(0)  # add batch dimension -> [1, T, C, H, W]
    frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)      # [B, C, T, H, W]
    return frames_tensor


# 4️⃣ Index page
def index(request):
    return render(request, "upload.html")

# 5️⃣ Video prediction
def predict(request):
    prediction = None
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        filepath = fs.path(filename)

        frames = video_to_frames(filepath)
        frames_tensor = preprocess_frames(frames)

        with torch.no_grad():
            outputs = model(frames_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        # Map numeric class to label
        class_labels = {0: "not theft", 1: "theft"}  # <-- update to your classes
        prediction = class_labels.get(predicted_class, "Unknown")

    return render(request, "upload.html", {"prediction": prediction})
