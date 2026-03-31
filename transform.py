import csv
import sys

input_file = "/Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/garment_extraction_lora/docs/review/one_shoulder_single_long_sleeve_caption_review.csv"

with open(input_file, mode='r', newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

for i in range(1, len(rows)):
    if len(rows[i]) < 6:
        continue
    row = rows[i]
    
    if row[4] == "viewer_side_left_right":
        row[4] = "garment_side_left_right"
    
    caption = row[5]
    # Use temporary placeholders
    caption = caption.replace("single long sleeve on left side, right shoulder bare", "TEMP_LEFT_SLEEVE")
    caption = caption.replace("single long sleeve on right side, left shoulder bare", "TEMP_RIGHT_SLEEVE")
    
    # Swap them
    caption = caption.replace("TEMP_LEFT_SLEEVE", "single long sleeve on right side, left shoulder bare")
    caption = caption.replace("TEMP_RIGHT_SLEEVE", "single long sleeve on left side, right shoulder bare")

    if row[0] == "asymmetric_exposed_bust_046.jpg":
        caption = caption.replace("one-shoulder crop top", "one-shoulder top")
    
    row[5] = caption

with open(input_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Transform complete.")
