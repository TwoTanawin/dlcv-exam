import torch
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
import cv2
from PIL import Image
import time
import os
import multiprocessing

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Myconfig:
    
    type_model_path = "/media/two-asus/Two-Works2/AIT/DLCV/final-exam/clssifier/model/v2-cloth-type/train-1400images-b6-10epochs-report/torch-efficientnet-b6-SGD-batch16-21-nov-23.pth"
    
    color_model_path = "/media/two-asus/Two-Works2/AIT/DLCV/final-exam/clssifier-color/model/v3/train-v3-1200images-notestset-b6-16-report/torch-cloth-color-efficientnet-b6-SGD-batch16-v3-22-nov-23.pth"

    folder_path = f'/media/two-asus/Two-Works2/AIT/DLCV/final-exam/combine-type-color/test/Colors'


# Define the transformation applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load type model
def load_eff_type_model(model_path):
    # load model
    model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=9)
    model.to(device)

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    
    return model

# load color model
def load_eff_color_model(model_path):
    # load model
    model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=8)
    model.to(device)

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    
    return model

# model prediction
def eff_classify(model, img_tensor):
    
    with torch.no_grad():
        model.eval()
        outputs = model(img_tensor)
    
    # Optionally, apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Move the probabilities tensor back to CPU for further processing (if needed)
    probabilities = probabilities.cpu()
    
    max_confident = torch.max(probabilities)
    
    return probabilities, max_confident

def image_process(file_path):
    # Step 4: Load and transform the image
    img = cv2.imread(file_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV reads images in BGR, so convert to RGB
    img_pil = Image.fromarray(img)  # Convert the NumPy array to a PIL Image
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # Move the image tensor to the GPU
    
    return img, img_tensor

def main():
    
    target_type = ["bagpack", "dress" , "longskirt", "longsleeveoutwear", "longsleevetop", "shorts","shortsleevetop", "shortsskirt", "trousers"]
    
    target_color = ["black", "blue", "cyan", "green", "purple", "red", "white", "yellow"]
    
    destination_path = "/media/two-asus/Two-Works2/AIT/DLCV/final-exam/combine-type-color/output/type/"
    
    inx = 0
    
    data_path = Myconfig()

    model_type = load_eff_type_model(data_path.type_model_path)
    
    model_color = load_eff_color_model(data_path.color_model_path)


    # List all files in the folder
    files = os.listdir(data_path.folder_path)
    
    file_path = "/media/two-asus/Two-Works2/AIT/DLCV/final-exam/combine-type-color/report/"+"cloth_color_report.text"
    file = open(file_path, "w")
    
    file.write("Cloth color Report:" + "\n")
    file.write("\n") 
    
    file.write("Tanawin Siriwan st123975 MMI" + "\n")
    
    file.write("\n") 
    
    file.write("Cloth tpye model efficientnetB6 : " + "\n")
    file.write("Cloth color model efficientnetB6 : " + "\n")
    # efficientnet
    file.write("\n") 
    

    # Loop through each file and read its contents
    for file_name in files:
        file_path = os.path.join(data_path.folder_path, file_name)
        
        inx += 1
        
        rec_filname = file_name
        
        
        img, img_tensor = image_process(file_path)


        probabilities_type, max_confident_type = eff_classify(model_type, img_tensor)
        
        probabilities_color, max_confident_color = eff_classify(model_color, img_tensor)
        
        # print(probabilities)
        
        # type probabilities
        
        print(f"type max confinent : {max_confident_type:.2f}")
        # print(max_confident)

        predicted_class_type = torch.argmax(probabilities_type).item()
        print(f"type Predicted class: {predicted_class_type}")
        
        print("-------------------------------------------")
        
        # color probabilities
        
        print(f"color max confinent : {max_confident_color:.2f}")
        # print(max_confident)

        predicted_class_color = torch.argmax(probabilities_color).item()
        print(f"color Predicted class: {predicted_class_color}")
        
        print("-------------------------------------------")
        
        for type_index in range(9):
            if predicted_class_type == type_index:
                target_type_str = str(target_type[type_index])
                for color_index in range(8):
                    if predicted_class_color == color_index:
                        target_color_str = str(target_color[color_index])
                        file_name = f"img_{target_type_str}_{max_confident_type:.2f}_{target_color_str}_{max_confident_color:.2f}_{inx}.jpg"
                        # cv2.imwrite(destination_path + f"{target_type_str}/{file_name}", img)
                        
                        # break
                        file.write(f"{rec_filname}, Prediction type: {target_type_str} confident: {max_confident_type:.2f}, Prediction color: {target_color_str} confident: {max_confident_color:.2f}, index: {inx}" + "\n")
        
                        end_time = time.time()
        
                        elapsed_time = end_time - start_time
                        
    file.write("\n")    
    file.write(f"Time taken: {elapsed_time:.2f} seconds" + "\n")

    file.close()

        
    
if __name__=="__main__":
    
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time

    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    print("Done...!")
