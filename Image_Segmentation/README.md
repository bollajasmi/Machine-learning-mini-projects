# Image Segmentation using K-Means Clustering

## Description
This project uses the K-Means clustering algorithm to segment a grayscale image into different regions. It groups similar pixels together and creates a new segmented image.

## Input
- Grayscale image (JPG/PNG format)
- Image is converted into pixel values between 0 and 1

## Output
- Segmented image with clustered pixels
- Output image is displayed and saved as a file (lung_kmeans.gif)

## Technologies Used
- Python
- NumPy
- Scikit-learn
- Matplotlib
- skimage

## Steps
1. Load the image
2. Convert image to grayscale and normalize
3. Reshape image into 1D array
4. Apply K-Means clustering
5. Assign cluster labels to pixels
6. Reshape and reconstruct the image
7. Display and save the output image

## Algorithm
- K-Means Clustering

## How to Run
1. Install required libraries:
   pip install numpy matplotlib scikit-learn scikit-image
2. Provide the image path in the code
3. Run the Python file or Jupyter Notebook

## Result
- The image is divided into 3 clusters (based on pixel intensity)
- Helps in identifying different regions in the image

## Conclusion
K-Means clustering is useful for image segmentation and helps in grouping similar pixels effectively.
Run the Python file or Jupyter Notebook
Result
The image is divided into 3 clusters (based on pixel intensity)
Helps in identifying different regions in the image
Conclusion
K-Means clustering is useful for image segmentation and helps in grouping similar pixels effectively.
