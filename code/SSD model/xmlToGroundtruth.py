# python3
# -- coding: utf-8 --
import numpy as np
import xml.etree.ElementTree as ET
import os
xmlDir = '../test-images'
classname = ['Mask','Face']


def listdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

xmlDir2 = listdirInMac(xmlDir)

for xmlName in xmlDir2:
    if xmlName.endswith('.xml'):
        xmlPath = os.path.join(xmlDir, xmlName)
        xmlName = xmlName.replace('.xml', '')
        print(xmlPath)
        output_info = []
        et = ET.parse(xmlPath)
        element = et.getroot()

        # 获取所有object对象
        element_objs = element.findall('object')
        # 图片文件名
        element_filename = element.find('filename').text
        print('element_filename:', element_filename)

        if len(element_objs) > 0:
            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                print('class_name:', class_name)
                if class_name=='face_mask':
                    class_id = 0
                elif class_name=='face':
                    class_id = 1

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                # print('(x1,y1,x2,y2):', (x1, y1, x2, y2))
                output_info.append([classname[class_id], x1, y1, x2, y2])
                print(output_info)

        file = open('../mAP/samples/test10/groundtruth/' + xmlName + '.txt', 'w')
        for i in range(len(output_info)):
            s = str(output_info[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',').replace('[','').replace(']', '').replace(',', '') + '\n'
            file.write(s)
        file.close()
