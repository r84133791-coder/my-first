import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from google.colab import files
uploaded = files.upload() # ይህንን ሲያስኬዱ "Choose Files" የሚል ቁልፍ ይመጣል፣ ምስልዎን ይምረጡ

# አንዴ ከተጫነ በኋላ የፋይሉን ስም ልክ እንደዚህ መቀበል ይችላሉ፡
import io
from PIL import Image

for filename in uploaded.keys():
    image_path = filename

input_image = Image.open(image_path)

# 1. ቀድሞ የሰለጠነ ሞዴል መጫን (Pre-trained ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval() # ሞዴሉን ለሙከራ (Evaluation) ሁኔታ እናስተካክለዋለን

# 2. የምስል ዝግጅት (Image Preprocessing)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# 3. ምስል መክፈት (from path) - 
# ከፊት 'r' ማድረጉ ወይም '\\' መጠቀም የዊንዶውስ ዱካ ስህተትን ያስወግዳል
image_path = r"c:\Users\Hp\Pictures\cange\OIP (1).jpg"
input_image = Image.open(image_path)

# ምስሉን ወደ RGB መቀየር (አንዳንድ ፐርሰንት ምስሎች PNG ወይም RGBA ሊሆኑ ስለሚችሉ 3 ቻናል እንዲኖራቸው ይረዳል)
input_image = input_image.convert("RGB")

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # ባች (Batch) ዴይሜንሽን እንጨምራለን

# 4. ትንበያ መስራት (Prediction)
with torch.no_grad():
    output = model(input_batch)

# 5. ውጤቱን ማየት
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print("የትንበያ ውጤት (Shape):", output.shape)
print("የተሳካ ትንበያ ተሰርቷል!")


