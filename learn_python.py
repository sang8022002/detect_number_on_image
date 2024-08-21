
import cv2
import pytesseract

# Chỉ định đường dẫn tới tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn đến tệp hình ảnh
image_path = 'image.png'

# Đọc hình ảnh bằng OpenCV
image = cv2.imread(image_path)

# detect text yellow color area
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
text = pytesseract.image_to_string(mask, config='--psm 6')
print(f'Detected text: {text}')



# Tính chiều rộng và chiều cao của ảnh
height, width, _ = image.shape

# Chia ảnh thành hai nửa: trái và phải
left_half = image[:, :width//2]
right_half = image[:, width//2:]
hsv_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2HSV)
hsv_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
# Nhận diện văn bản từ nửa trái
mask_left = cv2.inRange(hsv_left, lower_yellow, upper_yellow)
text_left = pytesseract.image_to_string(mask_left, config='--psm 6')
print(f'Detected text on the left side:\n{text_left}')

# Nhận diện văn bản từ nửa phải
mask_right = cv2.inRange(hsv_right, lower_yellow, upper_yellow)
text_right = pytesseract.image_to_string(mask_right, config='--psm 6')
print(f'Detected text on the right side:\n{text_right}')