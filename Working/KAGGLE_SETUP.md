# Kaggle Dataset Setup Guide

This guide will help you set up the Kaggle welding defect dataset for training.

## Dataset Information

- **Dataset**: sukmaadhiwijaya/welding-defect-object-detection
- **URL**: https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection

## Prerequisites

1. A Kaggle account (free): https://www.kaggle.com
2. Kaggle API credentials

## Step 1: Install Kaggle API

The Kaggle API is already in `requirements.txt` and should be installed. If not:

```bash
source venv/bin/activate
pip install kaggle
```

## Step 2: Set Up Kaggle Credentials

1. **Get your API token:**
   - Go to https://www.kaggle.com/account
   - Scroll down to the "API" section
   - Click "Create New API Token"
   - This will download a file called `kaggle.json`

2. **Place the credentials file:**
   ```bash
   # Create the .kaggle directory if it doesn't exist
   mkdir -p ~/.kaggle
   
   # Move the downloaded kaggle.json file
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   
   # Set proper permissions (required by Kaggle)
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Step 3: Download and Organize Dataset

Run the setup script:

```bash
source venv/bin/activate
python setup_kaggle_dataset.py
```

This script will:
1. Authenticate with Kaggle using your credentials
2. Download the welding defect dataset
3. Organize it into YOLOv8 format
4. Create train/val/test splits
5. Generate the dataset configuration file

## Step 4: Verify Dataset

After running the script, check that the dataset is organized:

```bash
# Check images
ls data/images/train/ | head -10
ls data/images/val/ | head -10

# Check labels
ls data/labels/train/ | head -10

# Check dataset config
cat data/weld_dataset.yaml
```

## Step 5: Train the Model

Once the dataset is set up, you can train:

```bash
source venv/bin/activate
python main.py --mode train
```

## Troubleshooting

### "403 Forbidden" Error
- Make sure you've accepted the dataset's terms of use on Kaggle
- Go to the dataset page and click "I Understand and Accept"

### "No module named 'kaggle'"
- Install: `pip install kaggle`
- Make sure virtual environment is activated

### "Could not find kaggle.json"
- Make sure the file is at `~/.kaggle/kaggle.json`
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Dataset structure issues
- The script tries to auto-detect the dataset structure
- If it fails, you may need to manually organize the files
- Check the `kaggle_download/` folder to see the original structure

## Notes

- The dataset will be downloaded to `kaggle_download/` folder
- Organized data will be in `data/` folder
- The script handles common dataset structures automatically
- You may need to adjust class IDs in labels if they don't match our categories

