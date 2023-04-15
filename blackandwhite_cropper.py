import cv2
import numpy as np
import os

def remove_border(img, threshold_black=10, threshold_white=245):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # For black borders
    _, thresh_black = cv2.threshold(gray, threshold_black, 255, cv2.THRESH_BINARY)
    contours_black, _ = cv2.findContours(thresh_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_black) > 0:
        cnt_black = max(contours_black, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt_black)
        img = img[y:y+h, x:x+w]

    # For white borders
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_white = cv2.threshold(gray, threshold_white, 255, cv2.THRESH_BINARY_INV)
    contours_white, _ = cv2.findContours(thresh_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_white) > 0:
        cnt_white = max(contours_white, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt_white)
        img = img[y:y+h, x:x+w]

    return img

folder_path = '/mnt/f/SCRAPED_DATA/Sorted/plunteere'
output_folder = '/mnt/f/SCRAPED_DATA/Sorted/plunteere_cropped'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        try:
            img = cv2.imread(os.path.join(folder_path, file_name))
            
            if img is None:
                print(f"Unable to read image: {file_name}")
                continue

            cropped_img = remove_border(img)

            output_file_name = os.path.join(output_folder, f"cropped_{file_name}")
            cv2.imwrite(output_file_name, cropped_img)
            print(f"Successfully processed: {file_name}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print("Finished processing all images.")