import os
import pandas as pd
# import tqdm
from PIL import Image
import zipfile

valid_df = pd.read_csv('origin/valid.csv')
# test_df = pd.read_csv('orgin/test.csv')
valid_df.head()
cate_id_map = {87: 0, 1034: 1, 131: 2, 318: 3, 588: 4}
PRED_PATH = "E:/holiday/cow/yolov5_cowboy/runs/detect/exp/labels"
IMAGE_PATH = "E:/holiday/cow/cowboyoutfits/val"

# list our prediction files path
prediction_files = os.listdir(PRED_PATH)
print('Number of test images with detections: ', len(prediction_files))


# convert yolo to coco annotation format
def yolo2cc_bbox(img_width, img_height, bbox):
    x = (bbox[0] - bbox[2] * 0.5) * img_width
    y = (bbox[1] - bbox[3] * 0.5) * img_height
    w = bbox[2] * img_width
    h = bbox[3] * img_height
    return (x, y, w, h)


# reverse the categories numer to the origin id
re_cate_id_map = dict(zip(cate_id_map.values(), cate_id_map.keys()))

print(re_cate_id_map)


def make_submission(df, PRED_PATH, IMAGE_PATH):
    output = []
    # for i in tqdm(range(len(df))):
    for i in range(len(df)):
        print(i)
        row = df.loc[i]
        image_id = row['id']
        file_name = row['file_name'].split('.')[0]
        if f'{file_name}.txt' in prediction_files:
            img = Image.open(f'{IMAGE_PATH}/{file_name}.jpg')
            width, height = img.size
            with open(f'{PRED_PATH}/{file_name}.txt', 'r') as file:
                for line in file:
                    preds = line.strip('\n').split(' ')
                    preds = list(map(float, preds))  # conver string to float
                    print(preds)
                    print(preds[1:-1])
                    cc_bbox = yolo2cc_bbox(width, height, preds[1:-1])
                    result = {
                        'image_id': image_id,
                        'category_id': re_cate_id_map[preds[0]],
                        'bbox': cc_bbox,
                        'score': preds[-1]
                    }

                    output.append(result)
    return output


sub_data = make_submission(valid_df, PRED_PATH, IMAGE_PATH)

op_pd = pd.DataFrame(sub_data)

op_pd.sample(10)


op_pd.to_json('cow_answer/answer.json', orient='records')
zf = zipfile.ZipFile('cow_answer/sample_answer.zip', 'w')
zf.write('cow_answer/answer.json', 'answer.json')
zf.close()