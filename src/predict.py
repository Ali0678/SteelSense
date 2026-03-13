import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
from model import SteelCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def predict_image(image_path, model_path):
    transform = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(image_path):
        print(f'Error: Image not found at {image_path}')
        return
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = SteelCNN(num_classes = 6).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location = DEVICE))
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim = 1)
        top_prob, top_class_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASSES[top_class_idx.item()]
    confidence = top_prob.item() * 100

    print(f'Analysis for: {os.path.basename(image_path)}')
    print(f'Prediction: {predicted_class.upper()}')
    print(f'Confidence: {confidence:.2f}%')

    return predicted_class

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, '..', 'models', 'best_model.pth')

    test_image = os.path.join(script_dir, '..', 'data', 'NEU_Clean', 'scratches', 'scratches_55.jpg')
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    predict_image(test_image, model_file)

