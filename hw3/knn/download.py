import urllib.request
import os
import time
url = "http://jwbinfosys.zju.edu.cn/CheckCode.aspx"
start = 0
img_num = 1000
for i in range(start, img_num):
    try:
        html_doc = urllib.request.urlopen(url)
        img = html_doc.read()
        with open('./res/' + str(i) + '.gif', 'wb') as f:
            f.write(img)
    except(Exception):
        i -= 1
    time.sleep(1)