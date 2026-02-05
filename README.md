# Traffic Sign Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) to classify traffic signs using the GTSRB dataset. It achieves **>95% accuracy** on the test set.

## Project Structure

- `dataset/`: Contains the training images (`Train/`) and test metadata (`Test.csv`, `Meta.csv`).
- `train_model.py`: Script to build, train, and evaluate the CNN model.
- `predict.py`: Script to run predictions on new images or random test images.
- `traffic_sign_cnn.pth`: Saved PyTorch model (generated after training).

## Requirements

Ensure you have Python installed with the following libraries:

```bash
pip install torch torchvision pandas numpy opencv-python scikit-learn pillow
or
pip install -r requirements.txt
```

## How to Train

To train the model from scratch, run:

```bash
python train_model.py
```

This will:

1. Load images from `dataset/Train` (cached in RAM for speed).
2. Train the CNN for 5 epochs.
3. specific validation accuracy after each epoch.
4. Save the trained model to `traffic_sign_cnn.pth`.
5. Evaluate the final accuracy on `dataset/Test.csv`.

**Note:** Training takes about 5-10 minutes on a CPU.

## How to Test / Predict

You can use `predict.py` to see the model in action.

### 1. Test a Random Image

Run the script without arguments to pick a random image from the `dataset/Test` folder and verify the prediction:

```bash
python predict.py
```

### 2. Test a Specific Image

Provide the path to an image file:

```bash
python predict.py path/to/image.png

example:
python predict.py "dataset/Test/00000.png"
```

## Model Architecture

The model is a 3-layer Convolutional Neural Network (CNN):

- **Conv1**: 32 filters, 5x5 kernel
- **Conv2**: 64 filters, 3x3 kernel
- **Conv3**: 128 filters, 3x3 kernel
- **FC Layers**: Dense layer (512 units) -> Output (43 classes)
