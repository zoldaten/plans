import cv2
import numpy as np
import json
import math

filename = '2.png'
min_distance_between_corners = 50  


image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, 100, 200)

corners = cv2.goodFeaturesToTrack(edges, maxCorners=200, qualityLevel=0.01, minDistance=5)


corners_list = []
if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        corners_list.append([int(x), int(y)])


filtered_corners = []

for current_corner in corners_list:
    keep = True
    for prev_corner in filtered_corners:
        dist = math.dist(current_corner, prev_corner)
        if dist < min_distance_between_corners:
            keep = False
            break
    if keep:
        filtered_corners.append(current_corner)


lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)


lines_list = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_list.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})


image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_color, (x1, y1), (x2, y2), (0, 255, 0), 2)


for corner in filtered_corners:
    x, y = corner
    cv2.circle(image_color, (x, y), 5, (0, 0, 255), -1)

cv2.imwrite('out.jpg',image_color)
#cv2.imshow('Обнаруженные линии и углы (отфильтрованные)', image_color)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


walls = []
for idx, line in enumerate(lines_list):
    wall = {
        'id': f"w{idx+1}",
        'points': [
            [line['x1'], line['y1']],
            [line['x2'], line['y2']]
        ]
    }
    walls.append(wall)

result = {
    "meta": { "source": filename },
    "walls": walls
}

json_output = json.dumps(result, indent=4, ensure_ascii=False)
#print(json_output)
with open("output_string.json", "w") as f:
        f.write(json_output)
