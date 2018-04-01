import sys
from PIL import Image
import re

def convert_annotations_to_csv(annotations_path, image_dir_path, csv_path):
    with open(annotations_path, 'r') as annotations_file, open(csv_path, 'w') as csv_file:
        annotations = annotations_file.readlines()
        annotations = [x.strip() for x in annotations]
        csv_file.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
        for line in annotations:
            print('------------')
            # line = line.decode('utf-8')
            
            coords = line.split(r'[')
            image_name = coords[0].split(':')[0]
            image_path = image_dir_path + image_name
            image_width, image_height = Image.open(open(image_path,'rb')).size            
            print(image_name, image_width, image_height)
            for coord in coords[1:]:
                # format of each line is xmin,ymin,width,height,color
                coord = coord.split(r']')[0]
                xmin = coord.split(',')[0]
                ymin = coord.split(',')[1]
                width = coord.split(',')[2]
                height = coord.split(',')[3]
                color = coord.split(',')[4]
                # print(xmin,ymin,width,height,color)
                xmax = int(xmin) + int(width)
                ymax = int(ymin) + int(height)
                csv_file.write('{},{},{},{},{},{},{},{}\n'.format(image_name, image_width, image_height, color, xmin, ymin, xmax, ymax))
    
def main():
    if len(sys.argv) != 4:
        print('usage is: python annotations_to_csv.py <annotations_path> <image_dir_path> <csv_path>')
    else:
        convert_annotations_to_csv(sys.argv[1], sys.argv[2], sys.argv[3])
    

if __name__ == '__main__':
    sys.exit(main())