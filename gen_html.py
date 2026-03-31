import csv
import os

csv_path = "/Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/garment_extraction_lora/docs/review/one_shoulder_single_long_sleeve_caption_review.csv"
image_dir = "/Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/garment_extraction_lora/data/target_images/one_shoulder_single_long_sleeve/"

html = ["<html><body><h1>Review Images</h1><table border='1'><tr><th>Image</th><th>File Name</th><th>Current Caption</th></tr>"]

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = os.path.join(image_dir, row['file_name'])
        html.append(f"<tr><td><img src='{img_path}' width='200'></td><td>{row['file_name']}</td><td>{row['caption_train']}</td></tr>")

html.append("</table></body></html>")

with open("review_images.html", "w") as f:
    f.write("\n".join(html))

print("HTML generated at review_images.html")
