Image Segmentation using K-Means Clustering
Description
This project uses the K-Means clustering algorithm to segment a grayscale image into different regions. It groups similar pixels together and creates a new segmented image.

Input
Grayscale image (JPG/PNG format)
Image is converted into pixel values between 0 and 1
Output
Segmented image with clustered pixels
Output image is displayed and saved as a file (lung_kmeans.gif)
Technologies Used
Python
NumPy
Scikit-learn
Matplotlib
skimage
Steps
Load the image
Convert image to grayscale and normalize
Reshape image into 1D array
Apply K-Means clustering
Assign cluster labels to pixels
Reshape and reconstruct the image
Display and save the output image
Algorithm
K-Means Clustering
How to Run
Install required libraries: pip install numpy matplotlib scikit-learn scikit-image
Provide the image path in the code
Run the Python file or Jupyter Notebook
Result
The image is divided into 3 clusters (based on pixel intensity)
Helps in identifying different regions in the image
Conclusion
K-Means clustering is useful for image segmentation and helps in grouping similar pixels effectively.
