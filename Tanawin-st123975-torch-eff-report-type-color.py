import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
import os
from sklearn.metrics import classification_report, accuracy_score
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Myconfig:
    model_path = "/media/two-asus/Two-Works2/AIT/DLCV/final-exam/clssifier-color/model/v3/train-v3-1200images-notestset-b7-16-report/torch-cloth-color-efficientnet-b7-SGD-batch16-v3-22-nov-23.pth"
    test_data_directory = "/media/two-asus/Two-Works2/AIT/DLCV/final-exam/correct-data-color/Google-Image-Scraper/unseen-2"

# Define the transformation applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def loadmodel(model_path):
    # Load the pre-trained EfficientNet model
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=8)
    model.to(device)

    # Load the saved state dictionary
    state_dict = torch.load(model_path)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    
    return model

def create_folder(folder_path):
    try:
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")


def main():
    
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    
    path_data = Myconfig()
    
    # data precessing
    test_data = ImageFolder(path_data.test_data_directory, transform=transform)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)
    
    # load model
    model = loadmodel(path_data.model_path)
    
    input_string = path_data.model_path
    result_list = input_string.split("/")
    
    folder_to_create = f'/media/two-asus/Two-Works2/AIT/DLCV/final-exam/clssifier-color/report/v3/unseen2-ontestset-color-{result_list[-1]}'
    
    # create folder
    create_folder(folder_to_create)

    # Remove empty strings from the result list
    result_list = [item for item in result_list if item]

    print(result_list[-1])
    
    start_time = time.time()

    file_path = folder_to_create+"/"+result_list[-1]+"_f1_score.text"
    file = open(file_path, "w")

    # Generate predictions and collect true labels
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Create a list of class names
    class_names = test_data.classes

    # Generate the classification report
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    print(report)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {overall_accuracy}")

    file.write("Classification Report:" + "\n")
    file.write(report + "\n")
    file.write(f"Overall Accuracy: {overall_accuracy:.2f}" + "\n")
    
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Time taken: {elapsed_time:.2f} seconds")

    file.write(f"Time taken: {elapsed_time:.2f} seconds" + "\n")

    file.close()

    # Create a list of class names
    # class_names = test_data.classes

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display the confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cm_display.plot()

    # Show the plot
    plt.title(result_list[-1])
    plt.savefig(f"{folder_to_create}/{result_list[-1]}.jpg")
    plt.show()
    
if __name__ == "__main__":
    
    main()
    
    print("Done...!")