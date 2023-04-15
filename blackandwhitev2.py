import cv2
import numpy as np
import os

def remove_border(img, threshold=50):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the average color of the border
    border_color = int((np.average(gray[0, :]) + np.average(gray[-1, :]) + np.average(gray[:, 0]) + np.average(gray[:, -1])) / 4)

    # Create a threshold based on the border color
    if border_color > 127:
        _, thresh = cv2.threshold(gray, border_color - threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, border_color + threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove the largest contour
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        img = img[y:y+h, x:x+w]

    return img

folder_path = '/mnt/f/SCRAPED_DATA/Unsorted/Alexander_Wells'
output_folder = '/mnt/f/SCRAPED_DATA/Unsorted/Alexander_Wells_Cropped'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        try:
            img = cv2.imread(os.path.join(folder_path, file_name))
            
            if img is None:
                print(f"Unable to read image: {file_name}")
                continue

            cropped_img = remove_border(img)  # Change this line

            output_file_name = os.path.join(output_folder, f"cropped_{file_name}")
            cv2.imwrite(output_file_name, cropped_img)
            print(f"Successfully processed: {file_name}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print("Finished processing all images.")