import cv2, os
root = "./00_data"
for fn in os.listdir(root):
    if fn.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(root, fn)
        img = cv2.imread(path)
        if img is None:
            print("❌ Failed to load:", path)
        else:
            print("✅ OK:", path)

