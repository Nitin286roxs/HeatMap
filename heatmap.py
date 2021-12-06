import json
import os
import cv2
import numpy as np
import sys
import random
import matplotlib
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.image import NonUniformImage
from scipy.signal import medfilt

from matplotlib import pyplot as plt
import base64
from scipy.stats import kde
import io
from io import StringIO
from PIL import Image

def _plot_heatmap(x, y, s, img_shape,bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins,range=[[0, img_shape[1]], [0, img_shape[0]]])
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def heatmap(DIC,img_string,camid,height,width):
    
    img=np.fromstring(img_string,np.uint8)
    I = img.reshape([height,width, 3])
    print(str(I.shape))
    print("height,width: "+str(height)+","+str(width))
    ########### Normal Case# ###########
    narr=np.array(DIC)
    x,y=narr.T
    img, extent = _plot_heatmap(x, y, 32,I.shape)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, extent=extent, cmap=cm.jet, aspect='auto')
    hmp_path = os.path.join("heatmap_{}.png".format("cam1"))
    fig.savefig(hmp_path)
    alpha = 0.6
    heatmap_img = cv2.imread(hmp_path)
    heatmap_img = cv2.resize(heatmap_img,(I.shape[1],I.shape[0]))
    #print(heatmap_img.shape)
    cv2.addWeighted(heatmap_img, alpha, I, 1 - alpha,0, I)
    overlayed_img_path = "heatmap_on_img.png"
    overlayed_img_path = os.path.join("./",overlayed_img_path)
    print("saving overlayed heatmap:{}".format(overlayed_img_path))
    cv2.imwrite(overlayed_img_path,I)
  
    img = plt.imread("heatmap_on_img.png")
    print(img.shape)
    
SampleImagePath=sys.argv[1]
img_arr=cv2.imread(SampleImagePath)
img_str=img_arr.tostring()
height=img_arr.shape[0]
width=img_arr.shape[1]
print(height,width)
filepath=sys.argv[2]
lines=[line.strip()for line in open(filepath,"r").readlines()]
li=[]	
for line in lines[1:]:
  xmin,xmax,ymin,ymax=map(int,line.strip().split(" "))
  cx=(xmin+xmax)/2
  cy=(ymin+ymax)/2
  li.append([cx,cy])
heatmap(li,img_str,"CAM_1",height,width)
