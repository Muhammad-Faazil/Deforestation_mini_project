import kaggle
import os

# Authenticate using your Kaggle API key (make sure kaggle.json is in C:\Users\faazi\.kaggle)
kaggle.api.authenticate()

# Create folder if it doesn’t exist
os.makedirs(r"C:\Users\faazi\Desktop\Deforestation_CNN\data", exist_ok=True)

# Download and unzip the dataset
kaggle.api.dataset_download_files(
    'akileshga/deforestation-non-deforestation-area-analysis',
    path=r'C:\Users\faazi\Desktop\Deforestation_CNN\data',
    unzip=True
)

print("✅ Dataset downloaded and extracted successfully!")
