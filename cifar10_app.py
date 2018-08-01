# -*- coding:utf-8 -*-

# TODO: 传入一张图片，输出它的预测值


from datetime import datetime
import math
import time
import sys
import getopt
import os

import tensorflow as tf

def inputs(image_dir):
    

if __name__ == '__main__':
    inputfile = ""
    # 首先读取 argv
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "n:", ["name="])
    except getopt.GetoptError:
        print("Input valid!")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            inputfile = arg
        else:
            print("Unknown instruction!")
            sys.exit(2)
    image_dir = os.path.join(os.getcwd(), inputfile)
    if not os.path.exists(image_dir):
        print("Invalid file: " + image_dir)
        sys.exit(2)
    print("[+] Load file: " + image_dir + "success")
    # 接下来读入图片
    image = inputs(image_dir)
