# Edge Detection with Color Structure Tensor

## Description

This repository contains code for edge detection in color images using the Color Structure Tensor. The implementation is part of a Computer Vision course under the guidance of Professor Dr. Feliz Bunyak Ersoy.

## Usage

### Requirements
- Python 3
- OpenCV
- NumPy
- Matplotlib

### Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/KatelynVanDyke/edge-detection-with-color-structure-tensor.git
```

2. Navigate to the project directory:
```bash
cd edge-detection-with-color-structure-tensor
```

3. Adjust the image path in the code to point to your desired input image:
```python
# Load the image
image = cv2.imread('path/to/your/image.jpg')
```

4. Run the script:
```bash
python edge_detection_color_structure_tensor.py
```

## Folder Structure

- `data/`: Output folder for generated images.
- `images/`: Input images folder.

## Results

The code generates various output images, including derivatives, 2D Color Structure Tensor elements, trace of the tensor, and gradient magnitude. These images are saved in the `data/` folder.
