import numpy as np
import torch
import time
import os
import torch.nn as nn
from preprocess.features import wl,get_time, get_gdata,get_circuit_name1,feature_values1,get_features1,get_routability,node_features,matrix_cong,matrix_finout,matrix_cong_tagh,matrix_coord,finout,vinout_blks,finout_blks,fine_grained_normalizer,vpr_txt
# from train import train1
from train.train1 import train1,test1
# from models import cnn_time
from models.cnn_time import CNN_croute

from models.cnn_time2 import CNN_croute2
from models.cnn_time1 import CNN_croute1
from preprocess.dataset import CustomImageDataset

from torch.utils.data import Dataset, DataLoader
start = time.time()

# file_address="/content/drive/MyDrive/dataset/blif-res2/stratixiv_arch.timing.xml"
# file_address= "/media/saba/Untitled/cong_mh/vtr-verilog-to-routing-congestion_placement/P

# file_address ="/home/saba/verilog_projects/congestion/vtr-verilog-to-routing/res-aware/res-cong/stratixiv_arch.timing.xml"
# "/media/saba/Untitled/congest"
# "/home/saba/verilog_projects/res-msh/vtr-verilog-to-routing/cong-res3/"
file_address="/home/saba/DL/time pred/dataset/blif-res2/stratixiv_arch.timing.xml/"
# file_address ="//home/saba/DL/time pred/dataset/res-other-titan2/stratixiv_arch.timing.xml/"

# file_address="/home/saba/verilog_projects/congestion/vtr-verilog-to-routing/vtr_flow/benchmarks/blif/ss/diffeq"

# file_address ="/media/saba/Untitled/cong_mh/run/vtr-verilog-to-routing/cong-ress5/dataset/blif-res/stratixiv_arch.timing.xml/"
# file_address="/media/saba/Untitled/cong_mh/run/vtr-verilog-to-routing/cong-ress3/dataset/blif-res/stratixiv_arch.timing.xml"
# file_address= "/home/saba/DL/time pred/dataset/blif-res2/stratixiv_arch.timing.xml"

# "/home/saba/DL/time pred/dataset/blif-res/stratixiv_arch.timing.xml/"
# # "/content/drive/MyDrive/new/dataset/blif-res/stratixiv_arch.timing.xml/"
#  "/content/drive/MyDrive/new/dataset/res-vtr1/stratixiv_arch.timing.xml/"
#
time_r=[]
input_cnn=[]
circuit_names=[]
routability=[]
fpga_size=[]
gf1=[]


# vblif=["adder_tree_2L_004bits.pre-vpr.blif","adder_tree_3L_064bits.pre-vpr.blif","adder_tree_2L_005bits.pre-vpr.blif","adder_tree_3L_096bits.pre-vpr.blif","adder_tree_2L_006bits.pre-vpr.blif","adder_tree_3L_128bits.pre-vpr.blif","adder_tree_2L_007bits.pre-vpr.blif", "and_latch.pre-vpr.blif","adder_tree_2L_008bits.pre-vpr.blif", "boundtop.pre-vpr.blif","adder_tree_2L_009bits.pre-vpr.blif", "ch_intrinsics.pre-vpr.blif","adder_tree_2L_010bits.pre-vpr.blif", "cordic.pre-vpr.blif","adder_tree_2L_011bits.pre-vpr.blif", "diffeq1.pre-vpr.blif","adder_tree_2L_012bits.pre-vpr.blif", "diffeq2.pre-vpr.blif","adder_tree_2L_013bits.pre-vpr.blif", "LU8PEEng.pre-vpr.blif","adder_tree_2L_014bits.pre-vpr.blif", "Md5Core.pre-vpr.blif","adder_tree_2L_015bits.pre-vpr.blif", "mkPktMerge.pre-vpr.blif","adder_tree_2L_016bits.pre-vpr.blif", "mkSMAdapter4B.pre-vpr.blif","adder_tree_2L_017bits.pre-vpr.blif", "mult_115.pre-vpr.blif","adder_tree_2L_018bits.pre-vpr.blif", "mult_116.pre-vpr.blif","adder_tree_2L_019bits.pre-vpr.blif", "mult_117.pre-vpr.blif","adder_tree_2L_020bits.pre-vpr.blif", "mult_118.pre-vpr.blif","adder_tree_2L_021bits.pre-vpr.blif", "mult_119.pre-vpr.blif","adder_tree_2L_022bits.pre-vpr.blif", "mult_120.pre-vpr.blif","adder_tree_2L_023bits.pre-vpr.blif", "mult_121.pre-vpr.blif","adder_tree_2L_024bits.pre-vpr.blif", "mult_122.pre-vpr.blif","adder_tree_2L_028bits.pre-vpr.blif", "mult_123.pre-vpr.blif","adder_tree_2L_032bits.pre-vpr.blif", "mult_124.pre-vpr.blif","adder_tree_2L_048bits.pre-vpr.blif", "mult_125.pre-vpr.blif","adder_tree_2L_064bits.pre-vpr.blif", "mult_126.pre-vpr.blif","adder_tree_2L_096bits.pre-vpr.blif", "mult_127.pre-vpr.blif","adder_tree_2L_128bits.pre-vpr.blif", "mult_128.pre-vpr.blif","adder_tree_3L_004bits.pre-vpr.blif", "mult_4x4.pre-vpr.blif","adder_tree_3L_005bits.pre-vpr.blif", "mult_5x5.pre-vpr.blif","adder_tree_3L_006bits.pre-vpr.blif", "mult_6x6.pre-vpr.blif","adder_tree_3L_007bits.pre-vpr.blif", "mult_7x7.pre-vpr.blif","adder_tree_3L_008bits.pre-vpr.blif", "mult_8x8.pre-vpr.blif","adder_tree_3L_009bits.pre-vpr.blif", "mult_9x9.pre-vpr.blif","adder_tree_3L_010bits.pre-vpr.blif", "multiclock_output_and_latch.pre-vpr.blif","adder_tree_3L_011bits.pre-vpr.blif", "multiclock_reader_writer.pre-vpr.blif","adder_tree_3L_012bits.pre-vpr.blif", "multiclock_separate_and_latch.pre-vpr.blif","adder_tree_3L_013bits.pre-vpr.blif", "or1200.pre-vpr.blif","adder_tree_3L_014bits.pre-vpr.blif", "pipelined_fft_64.pre-vpr.blif","adder_tree_3L_015bits.pre-vpr.blif", "raygentop.pre-vpr.blif","adder_tree_3L_016bits.pre-vpr.blif", "reduction_layer.pre-vpr.blif","adder_tree_3L_017bits.pre-vpr.blif", "robot_rl.pre-vpr.blif","adder_tree_3L_018bits.pre-vpr.blif", "sha.pre-vpr.blif","adder_tree_3L_019bits.pre-vpr.blif", "single_ff.pre-vpr.blif","adder_tree_3L_020bits.pre-vpr.blif", "single_wire.pre-vpr.blif","adder_tree_3L_021bits.pre-vpr.blif", "softmax.pre-vpr.blif","adder_tree_3L_022bits.pre-vpr.blif", "spree.pre-vpr.blif","adder_tree_3L_023bits.pre-vpr.blif", "stereovision0.pre-vpr.blif","adder_tree_3L_024bits.pre-vpr.blif", "stereovision1.pre-vpr.blif","adder_tree_3L_028bits.pre-vpr.blif", "stereovision2.pre-vpr.blif","adder_tree_3L_032bits.pre-vpr.blif", "stereovision3.pre-vpr.blif","adder_tree_3L_048bits.pre-vpr.blif", "test.pre-vpr.blif"]
# "pipelined_fft_64.pre-vpr.blif",
# blif=["and_latch.pre-vpr.blif","mult_125.pre-vpr.blif","boundtop.pre-vpr.blif","mult_126.pre-vpr.blif","ch_intrinsics.pre-vpr.blif", "cordic.pre-vpr.blif","mult_128.pre-vpr.blif","diffeq1.pre-vpr.blif","diffeq2.pre-vpr.blif","LU8PEEng.pre-vpr.blif","multiclock_output_and_latch.pre-vpr.blif","multiclock_reader_writer.pre-vpr.blif","multiclock_separate_and_latch.pre-vpr.blif","mkSMAdapter4B.pre-vpr.blif","or1200.pre-vpr.blif","mult_120.pre-vpr.blif","mult_121.pre-vpr.blif","mult_122.pre-vpr.blif","mult_123.pre-vpr.blif","mult_124.pre-vpr.blif"]
# blif=["leon2_stratixiv_arch_timing.blif"]
# blif=["Reed_Solomon_stratixiv_arch_timing.blif"]
# blif=[ "carpat_stratixiv_arch_timing.blif", "CH_DFSIN_stratixiv_arch_timing.blif","EKF-SLAM_Jacobians_stratixiv_arch_timing.blif" ,"leon2_stratixiv_arch_timing.blif","random_stratixiv_arch_timing.blif","Reed_Solomon_stratixiv_arch_timing.blif"]
# vblif=["and_latch.pre-vpr.blif", "mult_120.pre-vpr.blif","boundtop.pre-vpr.blif","mult_4x4.pre-vpr.blif","ch_intrinsics.pre-vpr.blif", "multiclock_output_and_latch.pre-vpr.blif","cordic.pre-vpr.blif","pipelined_fft_64.pre-vpr.blif","diffeq2.pre-vpr.blif", "sha.pre-vpr.blif","LU8PEEng.pre-vpr.blif","single_wire.pre-vpr.blif","Md5Core.pre-vpr.blif","stereovision0.pre-vpr.blif","mkPktMerge.pre-vpr.blif", "stereovision3.pre-vpr.blif","mult_115.pre-vpr.blif"]
# blif=["diffeq1.pre-vpr.blif","diffeq2.pre-vpr.blif","Reed_Solomon_stratixiv_arch_timing.blif","LU8PEEng.pre-vpr.blif","des.blif", "alu4.blif", "apex4.blif", "bigkey.blif",  "diffeq.blif","clma.blif","misex3.blif", "seq.blif","dsip.blif","elliptic.blif","ex1010.blif","frisc.blif","pdc.blif","s38417.blif","s38584.1.blif","seq.blif","tseng.blif"]
# blif=["cordic.pre-vpr.blif" ,"cordic.pre-vpr.blif","ch_intrinsics.pre-vpr.blif","boundtop.pre-vpr.blif","and_latch.pre-vpr.blif","multiclock_separate_and_latch.pre-vpr.blif","multiclock_reader_writer.pre-vpr.blif","multiclock_output_and_latch.pre-vpr.blif","mult_120.pre-vpr.blif","mult_121.pre-vpr.blif","mult_122.pre-vpr.blif","mult_123.pre-vpr.blif","mult_124.pre-vpr.blif","mult_125.pre-vpr.blif","mult_126.pre-vpr.blif","mult_127.pre-vpr.blif","mult_128.pre-vpr.blif", "or1200.pre-vpr.blif"]
# blif = ["random_stratixiv_arch_timing.blif"]
# blif=["leon3mp_stratixiv_arch_timing.blif"]
# "clma.blif"
# "tseng.blif","diffeq.blif",","pdc.blif","frisc.blif"
# blif=["elliptic.blif"]
# blif=["elliptic.blif"]
# blif=["alu4.blif","elliptic.blif","des.blif","ex1010.blif","bigkey.blif","clma.blif","dsip.blif","misex3.blif","seq.blif","clock_aliases.blif","apex4.blif"]
# blif=["apex4.blif-"]
# "ex1010.blif",
blif=["alu4.blif","des.blif","s38584.1.blif","bigkey.blif","clma.blif","dsip.blif","misex3.blif","seq.blif","elliptic.blif","clock_aliases.blif","apex4.blif"]
# blif=["alu4.blif","bigkey.blif","apex4.blif","misex3.blif",]
# blif=["alu4.blif", "apex4.blif", "bigkey.blif","misex3.blif", "seq.blif","dsip.blif","ex1010.blif","seq.blif"]
# blif=["alu4.blif","bigkey.blif" ,"diffeq.blif","des.blif"]
# blif=["elliptic.blif"]
# blif=["diffeq"]
# blif=["CH_DFSIN_stratixiv_arch_timing.blif"]
# file_text = open("./dataset2.txt", "r").read() # read the file as a string
# blif = file_text.splitlines() # split the string into a list of lines
# print(list_of_strings) # print the list of strings
# 
# for i in blif:
#    print("'%s'"%i)
# exit()
# # vblif=
# # , "clock_aliases.blif"]
f = np.empty([200,len(blif),2],dtype=int)
# np.ones((5,), dtype=int)

# pl=[1,7,13,17,21,31,34]
# [1,7,13,17,21,31,34,46,48,50,53,54]
itry=[]
# liss=[21,22]
# # for ch in range(1,53):
# #   for name in range(len(blif)):

# timee=[1574,1758,1803,3762,3784,3864,3934,4059,8425,8463,8481,8540,8581,8604,8683,8697,8746,8756.00,8838,8944,8958,8961,9696,9794,9873,10053,10064,10111,10142,10153,10163,10165,12311,12378,12431,12450,12455,12491,12625,19593,20438,20454,20471,20532,20538,20549,20665,20670,20740,20955,21006,21523,24078]
# outt=[5152.17,5152.17,5520.18,5520.18,5520.18,5520.18,5152.17,5152.17,5520.18,8832.29,8832.29,8832.29,8832.29,5520.18,5520.18,5520.18,5520.18,5520.18,8832.29,5520.18,8832.29,8832.29,5520.18,5520.18,5152.17,5152.17,5152.17,5152.17,5152.17,8832.29,5152.17,8832.29,5888.19,5888.19,5888.19,5888.19,5888.19,5888.19,5888.19,6624.22,6624.22,6624.22,6624.22,5520.18,5520.18,5520.18,5520.18,6624.22,5520.18,5520.18,6624.22,5520.18,5520.18]
# ss=[39,39,42,40,40,40,63,63,98,155,155,155,155,83,83,83,98,98,155,83,155,155,83,83,66,66,66,66,66,155,66,155,216,216,216,216,216,216,216,231,231,231,231,176,193,176,193,231,176,193,231,193,176]
# #  print(ch)
 
#     i=0
#   # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))
#   # if(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt") == [1]):
#   # for i in range(1,5):

#   # for i in [1,4]:
#   #  range(1,2):
# for ch in range(1,54):
#   for name in range(len(blif)):
#     # try:
#       time_r.append(get_time(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/time.txt"))
#       routability.append(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))   
#       print(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i))
#       # print(i)
#       circuit = get_gdata(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i) + "/features/graph_features.txt")
#       a,b,c= get_features1(circuit)
#       # print(c) # print(a,b,c)

#       f[ch][name]=c
#       if(c == [9,7]):
#         print( blif[name])
#       fpga_size.append(c)
#       # print(c,i)
#       # gf1.append(a)
#       if(len(a) == 31):
#         # print(a)
#         # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#         a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#         # print("--------\n",a)
#         a.pop(0)
#         a.pop(0)
#         gf1.append(a)
#       else:
#         a.pop(0)
#         a.pop(0)
#         gf1.append(a)
#       # gf1.pop(0)
#       # blks_size.append(c)
#       circuit_names.append(b)
#     # except IndexError:
#     #   break
#     # except OSError:
#     #    break

input_cnn=[]
fin=[]
fout=[]
cong=[]
cong_tagh=[]
in_minx=[]
in_miny=[]
in_xmax=[]
in_ymax=[]
out_minx=[]
out_miny=[]
out_xmax=[]
out_ymax=[]

fin1=0
fout1=0
cong1=0
cong_tagh1=0
in_minx1=0
in_miny1=0
in_xmax1=0
in_ymax1=0
out_minx1=0
out_miny1=0
out_xmax1=0
out_ymax1=0

# av_mtx = np.zeros((54,11))
def get_cong(cong_tagh):
    
  avg_cong=0
  max=0
  max_width=10000

  for i in range(81):
    for j in range(60):
        # print(c)
        # print(cong_tagh[i][j])
        if(max < cong_tagh[i][j]):
          max =  cong_tagh[i][j]


  for i in range(81):
    for j in range(60):
        if(max_width < cong_tagh[i][j]):
          avg_cong+=cong_tagh[i][j]-max_width
  return avg_cong

# for i in range(500):
# for i in range(246):
for i in range(700):
  fin.append([])
  fout.append([])
  cong.append([])
  cong_tagh.append([])
  in_minx.append([])
  in_miny.append([])
  in_xmax.append([])
  in_ymax.append([])
  out_minx.append([])
  out_miny.append([])
  out_xmax.append([])
  out_ymax.append([])
cnt=-1



# 3606460,3606984,3619952,3623450,3631378,3636165,3640054,3647181,3651210,3658089,3660607,3660883,3662363,3665469,3828233

# 3804755.6184,3865486.2642,3865486.2642,3865486.2642,3804755.6184,3804755.6184,3804755.6184,3863938.9866,3868194,3865486.2642,3863938.9866,3863938.9866,3865486.2642,3865486.2642,3868194

# 2711,3805,3805,3805,2711,2711,2711,3427,7718,3805,3427,3427,3805,3805,7718


# blif=["des.blif", "alu4.blif", "apex4.blif", "bigkey.blif",  "diffeq.blif","clma.blif","misex3.blif", "seq.blif","LU8PEEng.pre-vpr.blif", "boundtop.pre-vpr.blif", "ch_intrinsics.pre-vpr.blif", "cordic.pre-vpr.blif", "diffeq1.pre-vpr.blif", "diffeq2.pre-vpr.blif"]
# blif=["des.blif", "alu4.blif", "apex4.blif", "bigkey.blif",  "diffeq.blif","clma.blif","misex3.blif", "seq.blif","dsip.blif","elliptic.blif","ex1010.blif","frisc.blif","pdc.blif","s38417.blif","s38584.1.blif","seq.blif","tseng.blif"]
# print(len(fpga_size))
ss=[]
wrl=[]
import random
shuf=[]
import PIL
from PIL import Image as im 
import matplotlib.pyplot as plt
import seaborn as sea
# for i in range(39):
#    shuf.append(i+1)
# # random.shuffle(shuf)
# for ch in range(2,51):
#   ch1=ch


  
for name in range(len(blif)):
  for ch in range(2,39):
            ch1=ch
            # ch=ch1          
            # ch+=50
            i=0
            print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
            routability.append(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))   
                
            time_r.append(get_time(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/time.txt"))
            f[ch][name]=[81,60]
            # f[ch][name]=[194,144]
            print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
            # for i in range(1,n+1):
            c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/coord.txt",f[ch][name])
            # ,fpga_size[i])
            print("------\n")
            # print(c1)
            in_minx[name]=torch.tensor(c1)
            in_minx1= max(in_minx1,in_minx[name].max())
            
            # in_minx[name] = in_minx[name] / in_minx.max()
            #  torch.nn.functional.normalize(in_minx[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            in_miny[name]=torch.tensor(c2)
            in_miny1 = max(in_miny1,in_minx[name].max())
            # in_miny[name] = in_miny[name]/in_miny.max()
            # torch.nn.functional.normalize(in_miny[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            in_xmax[name]=torch.tensor(c3)
            in_xmax1= max(in_xmax1,in_xmax[name].max())
            # in_xmax[name] = in_xmax[name] / in_xmax.max()
            # torch.nn.functional.normalize(in_xmax[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            #print(in_xmax[name]
            #.view(1,11,fpga_size[name][0],fpga_size[name][1]),in_xmax[name].shape,"oooo")
            in_ymax[name]=torch.tensor(c4)
            in_ymax1 = max(in_ymax1,in_ymax[name].max())
            # in_ymax[name] = in_ymax[name]/in_ymax.max()
            # torch.nn.functional.normalize(in_ymax[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            out_minx[name]=torch.tensor(c5)
            out_minx1 = max(out_minx1,out_minx[name].max())
            # out_minx[name] = out_minx[name]/out_minx.max()
            # torch.nn.functional.normalize(out_minx[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            out_miny[name]=torch.tensor(c6)
            out_miny1 = max(out_miny1,out_miny[name].max())
            # out_miny[name] = out_miny[name]/out_miny.max()
            # torch.nn.functional.normalize(out_miny[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            out_xmax[name]=torch.tensor(c7)
            out_xmax1 = max(out_xmax1,out_xmax[name].max())
            # out_xmax[name] = out_xmax[name]/out_xmax.max()
            # torch.nn.functional.normalize(out_xmax[name],dim=0,p=2)
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            out_ymax[name]=torch.tensor(c8)
            out_ymax1 = max(out_ymax1,out_ymax[name].max())
            # out_ymax[name] = out_ymax[name]/out_ymax.max()

            fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/nodef-fin.txt",f[ch][name]))
            fin1 =  max(fin1,fin[name].max())
            # data = fin[name].numpy()
            # ax = sea.heatmap(data, linewidth=0.5)
            # # plt.imshow(data, cmap='hot', interpolation='nearest')
            # # plt.imshow(data, cmap='BuPu')
            # plt.show()
            # fin[name] = torch.nn.functional.normalize(fin[name],dim=0,p=2)
            # fin[name] = fin[name]/fin[name].max()
            # ,fpga_size[i]))
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # fin.append(torch.tensor(matrix_finout(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/nodef-fin.txt",fpga_size[i]))
            #
            #.view(-1,fpga_size[name][0],fpga_size[name][1]))
            fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/nodef-fout.txt",f[ch][name]))
            fout1 = max(fout1,fout[name].max())
            # data = fout[name].numpy()
            # plt.imshow(data, cmap='PuBu')
            # plt.show()
            
            # fout[name] = torch.nn.functional.normalize(fout[name],dim=0,p=2)
            # fout[name] = fout[name]/fout[name].max()

            # ,fpga_size[i]))
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # cong.append(matrix_cong(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/cong.txt",fpga_size[i]))
            cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/cong_tagh.txt",f[ch][name]))
            ss.append(get_cong(cong_tagh[name]))
            cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
            # print(cong_tagh[name])
            # data = cong_tagh[name].numpy()
            # plt.imshow(data, cmap='BuPu')
            # plt.show()
            # PIL.Image.fromarray(data)
            # im.show()
      
            # saving the final output  
            # as a PNG file 
            # data.save('./gfg_dummy_pic.png')
            # exit()
            
            # av_mtx[ch][name] = get_cong(cong_tagh[name])
            # print(cong_tagh[name][1],av_mtx[ch][name])
            # exit()
            # cong_tagh[name]=torch.nn.functional.normalize(cong_tagh[name],dim=0,p=2)
            # cong_tagh[name]=cong_tagh[name]/cong_tagh[name].max()

            # ,fpga_size[i]))
            #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # input_cnn.append(cong_tagh[name])
            input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
            
            # input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
            print(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i))
            # print(i)
            circuit = get_gdata(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i) + "/features/graph_features.txt")
            a,b,c= get_features1(circuit)
            # print(c) # print(a,b,c)

            f[ch][name]=c
            if(c == [9,7]):
              print( blif[name])
            fpga_size.append(c)
            # print(c,i)
            # gf1.append(a)
            if(len(a) == 31):
              # print(a)
              # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
              a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
              # print("--------\n",a)
              a.pop(0)
              a.pop(0)
              a.append(get_cong(cong_tagh[name]))
              a.append(wl(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/vpr_stdout.log"))
              
              # a.append()
              gf1.append(a)
              
            else:
              a.pop(0)
              a.pop(0)
              a.append(get_cong(cong_tagh[name]))
              a.append(wl(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/vpr_stdout.log"))
              
              gf1.append(a)
            # gf1.pop(0)
            # blks_size.append(c)
            circuit_names.append(b)
            # ch=ch1
            # # ch +=50
            # print(ch)
            # print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )

            # print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
            # routability.append(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))   
                
            # time_r.append(get_time(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/time.txt"))
            # f[ch][name]=[81,60]
            # # f[ch][name]=[194,144]
            # print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
            # # for i in range(1,n+1):
            # c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/coord.txt",f[ch][name])
            # # ,fpga_size[i])
            # print("------\n")
            # # print(c1)
            # in_minx[name]=torch.tensor(c1)
            # in_minx1= max(in_minx1,in_minx[name].max())
            
            # # in_minx[name] = in_minx[name] / in_minx.max()
            # #  torch.nn.functional.normalize(in_minx[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # in_miny[name]=torch.tensor(c2)
            # in_miny1 = max(in_miny1,in_minx[name].max())
            # # in_miny[name] = in_miny[name]/in_miny.max()
            # # torch.nn.functional.normalize(in_miny[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # in_xmax[name]=torch.tensor(c3)
            # in_xmax1= max(in_xmax1,in_xmax[name].max())
            # # in_xmax[name] = in_xmax[name] / in_xmax.max()
            # # torch.nn.functional.normalize(in_xmax[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # #print(in_xmax[name]
            # #.view(1,11,fpga_size[name][0],fpga_size[name][1]),in_xmax[name].shape,"oooo")
            # in_ymax[name]=torch.tensor(c4)
            # in_ymax1 = max(in_ymax1,in_ymax[name].max())
            # # in_ymax[name] = in_ymax[name]/in_ymax.max()
            # # torch.nn.functional.normalize(in_ymax[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # out_minx[name]=torch.tensor(c5)
            # out_minx1 = max(out_minx1,out_minx[name].max())
            # # out_minx[name] = out_minx[name]/out_minx.max()
            # # torch.nn.functional.normalize(out_minx[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # out_miny[name]=torch.tensor(c6)
            # out_miny1 = max(out_miny1,out_miny[name].max())
            # # out_miny[name] = out_miny[name]/out_miny.max()
            # # torch.nn.functional.normalize(out_miny[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # out_xmax[name]=torch.tensor(c7)
            # out_xmax1 = max(out_xmax1,out_xmax[name].max())
            # # out_xmax[name] = out_xmax[name]/out_xmax.max()
            # # torch.nn.functional.normalize(out_xmax[name],dim=0,p=2)
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # out_ymax[name]=torch.tensor(c8)
            # out_ymax1 = max(out_ymax1,out_ymax[name].max())
            # # out_ymax[name] = out_ymax[name]/out_ymax.max()

            # fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/nodef-fin.txt",f[ch][name]))
            # fin1 =  max(fin1,fin[name].max())
            # # fin[name] = torch.nn.functional.normalize(fin[name],dim=0,p=2)
            # # fin[name] = fin[name]/fin[name].max()
            # # ,fpga_size[i]))
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # # fin.append(torch.tensor(matrix_finout(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/nodef-fin.txt",fpga_size[i]))
            # #
            # #.view(-1,fpga_size[name][0],fpga_size[name][1]))
            # fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/nodef-fout.txt",f[ch][name]))
            # fout1 = max(fout1,fout[name].max())
            # # vinout_blks(blif,81*60,fin,fout)
            # # exit()

            # # vinout_blks()
            # # fout[name] = torch.nn.functional.normalize(fout[name],dim=0,p=2)
            # # fout[name] = fout[name]/fout[name].max()

            # # ,fpga_size[i]))
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # # cong.append(matrix_cong(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/cong.txt",fpga_size[i]))
            # cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/cong_tagh.txt",f[ch][name]))
            # ss.append(get_cong(cong_tagh[name]))
            # cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
            
            # # av_mtx[ch][name] = get_cong(cong_tagh[name])
            # # print(cong_tagh[name][1],av_mtx[ch][name])
            # # exit()
            # # cong_tagh[name]=torch.nn.functional.normalize(cong_tagh[name],dim=0,p=2)
            # # cong_tagh[name]=cong_tagh[name]/cong_tagh[name].max()

            # # ,fpga_size[i]))
            # #.view(-1,fpga_size[name][0],fpga_size[name][1])
            # # input_cnn.append(cong_tagh[name])
            # input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
            # # print(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i))
            # # print(i)
            # circuit = get_gdata(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i) + "/features/graph_features.txt")
            # a,b,c= get_features1(circuit)
            # # print(c) # print(a,b,c)

            # f[ch][name]=c
            # # if(c == [9,7]):
            # #   print( blif[name])
            # fpga_size.append(c)
            # # print(c,i)
            # # gf1.append(a)
            # if(len(a) == 31):
            #   # print(a)
            #   # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
            #   a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
            #   # print("--------\n",a)
            #   a.pop(0)
            #   a.pop(0)
            #   a.append(get_cong(cong_tagh[name]))
            #   a.append(wl(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/vpr_stdout.log"))
            #   # print(wl(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/vpr_stdout.log"))
            #   gf1.append(a)
              
            # else:
            #   a.pop(0)
            #   a.pop(0)
            #   a.append(get_cong(cong_tagh[name]))
            #   a.append(wl(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/vpr_stdout.log"))
            #   # print(wl(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/vpr_stdout.log"))
            #   gf1.append(a)

            # # gf1.pop(0)
            # # blks_size.append(c)
            # circuit_names.append(b)

# blif=["seq.blif"]
# for ch in range(1,27):
#   for name in range(len(blif)):
  
  
#             i=0
#             print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )

#             print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
#             routability.append(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))   
                
#             time_r.append(get_time(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/time.txt"))
#             f[ch][name]=[81,60]
#             # f[ch][name]=[194,144]
#             print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
#             # for i in range(1,n+1):
#             c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/coord.txt",f[ch][name])
#             # ,fpga_size[i])
#             print("------\n")
#             # print(c1)
#             in_minx[name]=torch.tensor(c1)
#             in_minx1= max(in_minx1,in_minx[name].max())
            
#             # in_minx[name] = in_minx[name] / in_minx.max()
#             #  torch.nn.functional.normalize(in_minx[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             in_miny[name]=torch.tensor(c2)
#             in_miny1 = max(in_miny1,in_minx[name].max())
#             # in_miny[name] = in_miny[name]/in_miny.max()
#             # torch.nn.functional.normalize(in_miny[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             in_xmax[name]=torch.tensor(c3)
#             in_xmax1= max(in_xmax1,in_xmax[name].max())
#             # in_xmax[name] = in_xmax[name] / in_xmax.max()
#             # torch.nn.functional.normalize(in_xmax[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             #print(in_xmax[name]
#             #.view(1,11,fpga_size[name][0],fpga_size[name][1]),in_xmax[name].shape,"oooo")
#             in_ymax[name]=torch.tensor(c4)
#             in_ymax1 = max(in_ymax1,in_ymax[name].max())
#             # in_ymax[name] = in_ymax[name]/in_ymax.max()
#             # torch.nn.functional.normalize(in_ymax[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             out_minx[name]=torch.tensor(c5)
#             out_minx1 = max(out_minx1,out_minx[name].max())
#             # out_minx[name] = out_minx[name]/out_minx.max()
#             # torch.nn.functional.normalize(out_minx[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             out_miny[name]=torch.tensor(c6)
#             out_miny1 = max(out_miny1,out_miny[name].max())
#             # out_miny[name] = out_miny[name]/out_miny.max()
#             # torch.nn.functional.normalize(out_miny[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             out_xmax[name]=torch.tensor(c7)
#             out_xmax1 = max(out_xmax1,out_xmax[name].max())
#             # out_xmax[name] = out_xmax[name]/out_xmax.max()
#             # torch.nn.functional.normalize(out_xmax[name],dim=0,p=2)
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             out_ymax[name]=torch.tensor(c8)
#             out_ymax1 = max(out_ymax1,out_ymax[name].max())
#             # out_ymax[name] = out_ymax[name]/out_ymax.max()

#             fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/nodef-fin.txt",f[ch][name]))
#             fin1 =  max(fin1,fin[name].max())
#             # fin[name] = torch.nn.functional.normalize(fin[name],dim=0,p=2)
#             # fin[name] = fin[name]/fin[name].max()
#             # ,fpga_size[i]))
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             # fin.append(torch.tensor(matrix_finout(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/nodef-fin.txt",fpga_size[i]))
#             #
#             #.view(-1,fpga_size[name][0],fpga_size[name][1]))
#             fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/nodef-fout.txt",f[ch][name]))
#             fout1 = max(fout1,fout[name].max())
#             # fout[name] = torch.nn.functional.normalize(fout[name],dim=0,p=2)
#             # fout[name] = fout[name]/fout[name].max()

#             # ,fpga_size[i]))
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             # cong.append(matrix_cong(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/cong.txt",fpga_size[i]))
#             cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/cong_tagh.txt",f[ch][name]))
#             ss.append(get_cong(cong_tagh[name]))
#             cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
            
#             av_mtx[ch][name] = get_cong(cong_tagh[name])
#             # print(cong_tagh[name][1],av_mtx[ch][name])
#             # exit()
#             # cong_tagh[name]=torch.nn.functional.normalize(cong_tagh[name],dim=0,p=2)
#             # cong_tagh[name]=cong_tagh[name]/cong_tagh[name].max()

#             # ,fpga_size[i]))
#             #.view(-1,fpga_size[name][0],fpga_size[name][1])
#             # input_cnn.append(cong_tagh[name])
#             input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
#             print(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i))
#             # print(i)
#             circuit = get_gdata(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i) + "/features/graph_features.txt")
#             a,b,c= get_features1(circuit)
#             # print(c) # print(a,b,c)

#             f[ch][name]=c
#             if(c == [9,7]):
#               print( blif[name])
#             fpga_size.append(c)
#             # print(c,i)
#             # gf1.append(a)
#             if(len(a) == 31):
#               # print(a)
#               # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#               a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#               # print("--------\n",a)
#               a.pop(0)
#               a.pop(0)
#               a.append(get_cong(cong_tagh[name]))
#               gf1.append(a)
              
#             else:
#               a.pop(0)
#               a.pop(0)
#               a.append(get_cong(cong_tagh[name]))
#               gf1.append(a)
#             # gf1.pop(0)
#             # blks_size.append(c)
#             circuit_names.append(b)

# file_address ="/media/saba/Untitled/congest"
# file_text = open("./dataset.txt", "r").read() # read the file as a string
# blif = file_text.splitlines() 
# # # for ch in range(1,51):
# file_text = open("./dataset2.txt", "r").read() # read the file as a string
# blif = file_text.splitlines()
# print(blif)
# llist=[]
# llist1=[]
# lab=[]
# for name in range(len(blif)):
#   i=0
#   congee=[]
#   wll=[]

#   # print(blif[name])
#   # print(file_address + "/" + blif[name]  + "/vpr_stdout.log")
  
  
#   congee=[]
#   wll=[]
  
#   with open(file_address + "/" + blif[name]  + "/vpr_stdout.log") as f:   
#     cnt=0    
#     for line in f:
#         cnt+=1
#         if('Block Type   # Blocks' in line):
#            break
#   # print(cnt)
#   with open(file_address + "/" + blif[name]  + "/vpr_stdout.log") as f:   
#     cnt1=0    
#     for line in f: 
#        cnt1+=1
#        if(cnt == cnt1-5): 
#           lab.append(line.strip().split()[1])
#           # .split('          ')[1])
#           break  
#   # print(len(lab),name,lab, blif[name])
#   with open(file_address + "/" + blif[name]  + "/vpr_stdout.log") as f:       
#     for line in f:
#   #       if('## Placement Quench took' in line):
#   #          print(line)
#   #          break
#   #   print(line)
#       if("CONG LIMM " in line):
#         wll.append(float(line.strip().split(' ')[2].split(',')[1]))

#         congee.append(float(line.strip().split(' ')[2].split(',')[0]))
#   print(lab[name])
#   llist.append('lab : %s'%lab[name])
#   llist.append(sum(wll) / len(wll))
#   llist1.append('lab : %s'%lab[name])
#   llist1.append(sum(congee) / len(congee) )

  # line_number = 0
  # with open(file_address + "/" + blif[name]  + "/vpr_stdout.log") as f:
  #   for i, line in enumerate(f, 1):
  #     if "## Placement Quench took" in line:
  #         line_number = i
  #         print
  #     break
  #   for i, line in enumerate(f, 1):
  #     if( i == line_number - 1 ):
  #       print(line)
# get the previous line
# prev_line = linecache.getline("file.txt", line_number - 1)
# print(prev_line)

#         if("CONG LIMM " in line):
#             wll.append(float(line.strip().split(' ')[2].split(',')[1]))

#             congee.append(float(line.strip().split(' ')[2].split(',')[0]))
#   llist.append(sum(wll) / len(wll))
#   llist1.append(sum(congee) / len(congee) )
#   congee=[]
#   wll=[]

# print(llist)
# print('------------------------')
# print(llist1)


#             # print(line.strip().split(' ')[2].split(','))

# exit()
# for name in range(len(blif)):
  
  
#             # i=0
#             # print("---------------------------------")
#             # print(blif[name])
#             # print(file_address + "/" + blif[name]  + "/vpr_stdout.log")
#             # with open(file_address + "/" + blif[name]  + "/vpr_stdout.log") as f:       
#             #   for line in f:
#             #       if("CONG LIMM " in line):
#             #           print(line.strip().split(' ')[2].split(','))
#             # exit()
#             circuit = get_gdata(file_address + "/" + blif[name]  + "/features/graph_features.txt")
#             a,b,c= get_features1(circuit)
#             # if c==[81,60]:
#             if c==[23,17]:
#               # print(file_address + "/"+ blif[name] + "/" )

#               # print(file_address + "/"+ blif[name] + "/" )
#               routability.append(get_routability(file_address + "/" + blif[name]   +"/features/routability.txt"))   
                  
#               time_r.append(get_time(file_address + "/"+ blif[name]  + "/features/time.txt"))
#               # f[ch][name]=[81,60]
#               # f[ch][name]=[194,144]
#               print(file_address + "/"+ blif[name] + "/" )
#               # for i in range(1,n+1):
#               c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/"  +"/features/coord.txt",c)
#               # ,fpga_size[i])
#               print("------\n")
#               # print(c1)
#               in_minx[name]=torch.tensor(c1)
#               in_minx1= max(in_minx1,in_minx[name].max())
#               data = in_minx[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/in_minx.png")
#               # plt.show()
              
#               # in_minx[name] = in_minx[name] / in_minx.max()
#               #  torch.nn.functional.normalize(in_minx[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               in_miny[name]=torch.tensor(c2)
#               in_miny1 = max(in_miny1,in_minx[name].max())
#               data = in_miny[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/in_miny.png")
#               # plt.show()
#               # in_miny[name] = in_miny[name]/in_miny.max()
#               # torch.nn.functional.normalize(in_miny[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               in_xmax[name]=torch.tensor(c3)
#               in_xmax1= max(in_xmax1,in_xmax[name].max())
#               data = in_xmax[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/in_xmax.png")
#               # plt.show()
#               # in_xmax[name] = in_xmax[name] / in_xmax.max()
#               # torch.nn.functional.normalize(in_xmax[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               #print(in_xmax[name]
#               #.view(1,11,fpga_size[name][0],fpga_size[name][1]),in_xmax[name].shape,"oooo")
#               in_ymax[name]=torch.tensor(c4)
#               in_ymax1 = max(in_ymax1,in_ymax[name].max())
#               data = in_ymax[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/in_ymax.png")
#               # plt.show()
#               # in_ymax[name] = in_ymax[name]/in_ymax.max()
#               # torch.nn.functional.normalize(in_ymax[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_minx[name]=torch.tensor(c5)
#               out_minx1 = max(out_minx1,out_minx[name].max())
#               data = out_minx[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/out_minx.png")
#               # plt.show()
#               # out_minx[name] = out_minx[name]/out_minx.max()
#               # torch.nn.functional.normalize(out_minx[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_miny[name]=torch.tensor(c6)
#               out_miny1 = max(out_miny1,out_miny[name].max())
#               data = out_miny[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/out_miny.png")
#               # plt.show()
#               # out_miny[name] = out_miny[name]/out_miny.max()
#               # torch.nn.functional.normalize(out_miny[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_xmax[name]=torch.tensor(c7)
#               out_xmax1 = max(out_xmax1,out_xmax[name].max())
#               data = out_xmax[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/out_xmax.png")
#               # plt.show()
#               # out_xmax[name] = out_xmax[name]/out_xmax.max()
#               # torch.nn.functional.normalize(out_xmax[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_ymax[name]=torch.tensor(c8)
#               out_ymax1 = max(out_ymax1,out_ymax[name].max())
#               data = out_ymax[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/out_ymax.png")
#               # plt.show()
#               # out_ymax[name] = out_ymax[name]/out_ymax.max()

#               fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/"  +"/features/nodef-fin.txt",c))
#               fin1 =  max(fin1,fin[name].max())
              
#               data = fin[name].numpy()
#               # ax = sea.heatmap(data, linewidth=0.5)
#               # plt.imshow(data, cmap='hot', interpolation='nearest')
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/fin.png")
#               # plt.show()
#               # fin[name] = torch.nn.functional.normalize(fin[name],dim=0,p=2)
#               # fin[name] = fin[name]/fin[name].max()
#               # ,fpga_size[i]))
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               # fin.append(torch.tensor(matrix_finout(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/nodef-fin.txt",fpga_size[i]))
#               #
#               #.view(-1,fpga_size[name][0],fpga_size[name][1]))
#               fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/"  + "/features/nodef-fout.txt",c))
#               fout1 = max(fout1,fout[name].max())
#               # fout1 = max(fout1,fout[name].max())
#               data = fout[name].numpy()
#               # plt.imshow(data, cmap='viridis')
#               # plt.savefig(file_address + "/fout.png")
#               # plt.show()
#               # exit()
#               # fout[name] = torch.nn.functional.normalize(fout[name],dim=0,p=2)
#               # fout[name] = fout[name]/fout[name].max()

#               # ,fpga_size[i]))
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               # cong.append(matrix_cong(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/cong.txt",fpga_size[i]))
#               cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + "/features/cong_tagh.txt",c))
#               # ss.append(get_cong(cong_tagh[name]))
#               # print(cong_tagh)
#               cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
#               data = cong_tagh[name].numpy()
#               plt.imshow(data, cmap='viridis')
#               plt.savefig(file_address + "/cong_tagh.png")
#               # plt.show()
              
#               # av_mtx[ch][name] = get_cong(cong_tagh[name])
#               # print(cong_tagh[name][1],av_mtx[ch][name])
#               # exit()
#               # cong_tagh[name]=torch.nn.functional.normalize(cong_tagh[name],dim=0,p=2)
#               # cong_tagh[name]=cong_tagh[name]/cong_tagh[name].max()

#               # ,fpga_size[i]))
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               # input_cnn.append(cong_tagh[name])
#               input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
#               print(file_address + "/" + blif[name] )
#               # print(i)
              
#               # print(c) # print(a,b,c)

#               # c=c
#               if(c == [9,7]):
#                 print( blif[name])
#               fpga_size.append(c)
#               # print(c,i)
#               # gf1.append(a)
#               if(len(a) == 31):
#                 # print(a)
#                 # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#                 a.insert(0,get_routability(file_address + "/" + blif[name]  +"/features/cw.txt")[0])
#                 # print("--------\n",a)
#                 a.pop(0)
#                 a.pop(0)
#                 a.append(get_cong(cong_tagh[name]))
#                 a.append(wl(file_address + "/" + blif[name]+"/vpr_stdout.log"))
#                 gf1.append(a)
                
#               else:
#                 a.pop(0)
#                 a.pop(0)
#                 a.append(get_cong(cong_tagh[name]))
#                 a.append(wl(file_address + "/" + blif[name]+"/vpr_stdout.log"))
# #               
#                 gf1.append(a)
#               # gf1.pop(0)
#               # blks_size.append(c)
#               circuit_names.append(b)



  # for ch in range(,20):
  
  #           i=0
  #           print(file_address + "/"+ blif[name] + "/"+ "/" +str(i) )

  #           print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
  #           routability.append(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))   
                
  #           time_r.append(get_time(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/time.txt"))
  #           f[ch][name]=[81,60]
  #           # f[ch][name]=[81,60]
  #           print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
  #           # for i in range(1,n+1):
  #           c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/coord.txt",f[ch][name])
  #           # ,fpga_size[i])
  #           print("------\n")
  #           # print(c1)
  #           in_minx[name]=torch.tensor(c1)
  #           in_minx1= max(in_minx1,in_minx[name].max())
            
  #           # in_minx[name] = in_minx[name] / in_minx.max()
  #           #  torch.nn.functional.normalize(in_minx[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           in_miny[name]=torch.tensor(c2)
  #           in_miny1 = max(in_miny1,in_minx[name].max())
  #           # in_miny[name] = in_miny[name]/in_miny.max()
  #           # torch.nn.functional.normalize(in_miny[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           in_xmax[name]=torch.tensor(c3)
  #           in_xmax1= max(in_xmax1,in_xmax[name].max())
  #           # in_xmax[name] = in_xmax[name] / in_xmax.max()
  #           # torch.nn.functional.normalize(in_xmax[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           #print(in_xmax[name]
  #           #.view(1,11,fpga_size[name][0],fpga_size[name][1]),in_xmax[name].shape,"oooo")
  #           in_ymax[name]=torch.tensor(c4)
  #           in_ymax1 = max(in_ymax1,in_ymax[name].max())
  #           # in_ymax[name] = in_ymax[name]/in_ymax.max()
  #           # torch.nn.functional.normalize(in_ymax[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           out_minx[name]=torch.tensor(c5)
  #           out_minx1 = max(out_minx1,out_minx[name].max())
  #           # out_minx[name] = out_minx[name]/out_minx.max()
  #           # torch.nn.functional.normalize(out_minx[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           out_miny[name]=torch.tensor(c6)
  #           out_miny1 = max(out_miny1,out_miny[name].max())
  #           # out_miny[name] = out_miny[name]/out_miny.max()
  #           # torch.nn.functional.normalize(out_miny[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           out_xmax[name]=torch.tensor(c7)
  #           out_xmax1 = max(out_xmax1,out_xmax[name].max())
  #           # out_xmax[name] = out_xmax[name]/out_xmax.max()
  #           # torch.nn.functional.normalize(out_xmax[name],dim=0,p=2)
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           out_ymax[name]=torch.tensor(c8)
  #           out_ymax1 = max(out_ymax1,out_ymax[name].max())
  #           # out_ymax[name] = out_ymax[name]/out_ymax.max()

  #           fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/nodef-fin.txt",f[ch][name]))
  #           fin1 =  max(fin1,fin[name].max())
  #           # fin[name] = torch.nn.functional.normalize(fin[name],dim=0,p=2)
  #           # fin[name] = fin[name]/fin[name].max()
  #           # ,fpga_size[i]))
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           # fin.append(torch.tensor(matrix_finout(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/nodef-fin.txt",fpga_size[i]))
  #           #
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1]))
  #           fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/nodef-fout.txt",f[ch][name]))
  #           fout1 = max(fout1,fout[name].max())
  #           # fout[name] = torch.nn.functional.normalize(fout[name],dim=0,p=2)
  #           # fout[name] = fout[name]/fout[name].max()

  #           # ,fpga_size[i]))
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           # cong.append(matrix_cong(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/cong.txt",fpga_size[i]))
  #           cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/cong_tagh.txt",f[ch][name]))
  #           cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
  #           # av_mtx[ch][name] = get_cong(cong_tagh[name])
  #           # print(cong_tagh[name][1],av_mtx[ch][name])
  #           # exit()
  #           # cong_tagh[name]=torch.nn.functional.normalize(cong_tagh[name],dim=0,p=2)
  #           # cong_tagh[name]=cong_tagh[name]/cong_tagh[name].max()

  #           # ,fpga_size[i]))
  #           #.view(-1,fpga_size[name][0],fpga_size[name][1])
  #           input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
  #           print(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i))
  #           # print(i)
  #           circuit = get_gdata(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i) + "/features/graph_features.txt")
  #           a,b,c= get_features1(circuit)
  #           # print(c) # print(a,b,c)

  #           f[ch][name]=c
  #           if(c == [9,7]):
  #             print( blif[name])
  #           fpga_size.append(c)
  #           # print(c,i)
  #           # gf1.append(a)
  #           if(len(a) == 31):
  #             # print(a)
  #             # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
  #             a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
  #             # print("--------\n",a)
  #             a.pop(0)
  #             a.pop(0)
  #             gf1.append(a)
  #           else:
  #             a.pop(0)
  #             a.pop(0)
  #             gf1.append(a)
  #           # gf1.pop(0)
  #           # blks_size.append(c)
  #           circuit_names.append(b)

# , "clock_aliases.blif"]
# for ch in range(1,54):
#   for name in range(len(blif)):
#      av_mtx


# exit()

# itry=[]
# for ch in range(1,54):
#   l+=1
#   for name in range(len(blif)):
#       i=0
#       l+=1
#       # print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
#     # for i in range(1,5):
        
#       # if(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt") == [1]):
#       try:
#             print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
#             routability.append(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/routability.txt"))   
                
#             time_r.append(get_time(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/time.txt"))
            
#             if(os.stat(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/coord.txt").st_size != 0):
#               # print(blif[name], f[ch][name])
#               f[ch][name]=[81,60]
#             # if (f[ch][name][0] == 81):
#               # f[ch][name] = [81,60]
#               # print(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i) )
#               # for i in range(1,n+1):
#               c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/coord.txt",f[ch][name])
#               # ,fpga_size[i])
#               # print("------\n")
#               in_minx[name]=torch.tensor(c1)
#               in_minx1= max(in_minx1,in_minx[name].max())
              
#               # in_minx[name] = in_minx[name] / in_minx.max()
#               #  torch.nn.functional.normalize(in_minx[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               in_miny[name]=torch.tensor(c2)
#               in_miny1 = max(in_miny1,in_minx[name].max())
#               # in_miny[name] = in_miny[name]/in_miny.max()
#               # torch.nn.functional.normalize(in_miny[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               in_xmax[name]=torch.tensor(c3)
#               in_xmax1= max(in_xmax1,in_xmax[name].max())
#               # in_xmax[name] = in_xmax[name] / in_xmax.max()
#               # torch.nn.functional.normalize(in_xmax[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               #print(in_xmax[name]
#               #.view(1,11,fpga_size[name][0],fpga_size[name][1]),in_xmax[name].shape,"oooo")
#               in_ymax[name]=torch.tensor(c4)
#               in_ymax1 = max(in_ymax1,in_ymax[name].max())
#               # in_ymax[name] = in_ymax[name]/in_ymax.max()
#               # torch.nn.functional.normalize(in_ymax[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_minx[name]=torch.tensor(c5)
#               out_minx1 = max(out_minx1,out_minx[name].max())
#               # out_minx[name] = out_minx[name]/out_minx.max()
#               # torch.nn.functional.normalize(out_minx[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_miny[name]=torch.tensor(c6)
#               out_miny1 = max(out_miny1,out_miny[name].max())
#               # out_miny[name] = out_miny[name]/out_miny.max()
#               # torch.nn.functional.normalize(out_miny[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_xmax[name]=torch.tensor(c7)
#               out_xmax1 = max(out_xmax1,out_xmax[name].max())
#               # out_xmax[name] = out_xmax[name]/out_xmax.max()
#               # torch.nn.functional.normalize(out_xmax[name],dim=0,p=2)
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               out_ymax[name]=torch.tensor(c8)
#               out_ymax1 = max(out_ymax1,out_ymax[name].max())
#               # out_ymax[name] = out_ymax[name]/out_ymax.max()

#               fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  +"/features/nodef-fin.txt",f[ch][name]))
#               fin1 =  max(fin1,fin[name].max())
#               # fin[name] = torch.nn.functional.normalize(fin[name],dim=0,p=2)
#               # fin[name] = fin[name]/fin[name].max()
#               # ,fpga_size[i]))
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               # fin.append(torch.tensor(matrix_finout(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/nodef-fin.txt",fpga_size[i]))
#               #
#               #.view(-1,fpga_size[name][0],fpga_size[name][1]))
#               fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/nodef-fout.txt",f[ch][name]))
#               fout1 = max(fout1,fout[name].max())
#               # fout[name] = torch.nn.functional.normalize(fout[name],dim=0,p=2)
#               # fout[name] = fout[name]/fout[name].max()

#               # ,fpga_size[i]))
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               # cong.append(matrix_cong(file_address+  "Projects/dataset/stratixiv_arch.timing.xml" + "/" + ch +"/" + blif[name]+ "/features/cong.txt",fpga_size[i]))
#               cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + str(ch)+ "/" +str(i)  + "/features/cong_tagh.txt",f[ch][name]))
#               # if()
#               av_mtx[ch][name] = get_cong(cong_tagh[name])
#               # print("AVGG",get_cong(cong_tagh[name]))
#               # print(cong_tagh[name])
#               # print(cong_tagh)
#               # exit()
              
#               cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
#               # cong_tagh[name]=torch.nn.functional.normalize(cong_tagh[name],dim=0,p=2)
#               # cong_tagh[name]=cong_tagh[name]/cong_tagh[name].max()

#               # ,fpga_size[i]))
#               #.view(-1,fpga_size[name][0],fpga_size[name][1])
#               input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
#               # print(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i))
#               # print(i)
#               circuit = get_gdata(file_address + "/" + blif[name] + "/"+ str(ch) + "/" +str(i) + "/features/graph_features.txt")
#               a,b,c= get_features1(circuit)
#               # print(c) # print(a,b,c)

#               f[ch][name]=c
#               if(c == [9,7]):
#                 print( blif[name])
#               fpga_size.append(c)
#               # print(c,i)
#               # gf1.append(a)
#               if(len(a) == 31):
#                 # print(a)
#                 # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#                 a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
#                 # print("--------\n",a)
#                 a.pop(0)
#                 a.pop(0)
#                 gf1.append(a)
#               else:
#                 a.pop(0)
#                 a.pop(0)
#                 gf1.append(a)
#               # gf1.pop(0)
#               # blks_size.append(c)
#               circuit_names.append(b)
#       except IndexError:
#         break
#       except OSError:
#         break

res=[]

max_width = 76
# print(time_r[:51])

# print(time_r[51:])
# exit()
# max=0
#     for(int i=0;i<int(device_ctx.grid.width());i++){
#         for(int j=0;j<int(device_ctx.grid.height());j++){
#             if(max<cong_matrix_new[i][j]){
#                 max = cong_matrix_new[i][j];
#             }
#             // cong_matrix_new[i][j] = 0.0;
#         }
#     }

#     for(int i=0;i<int(device_ctx.grid.width());i++){
#         for(int j=0;j<int(device_ctx.grid.height());j++){
#             if(cong_matrix_new[i][j]>max_width){
#                 avg+=cong_matrix_new[i][j]-max_width;
#                 num+=1.0;
#             }
#         }
#     }
print(len(fpga_size))
# for i in cong_tagh[0]:
#    print(i)
# print(len(cong_tagh[0]))
# exit()
# max=[]
# avg_cong=[]
# print(len(cong_tagh[0]))
# # print(torch.tensor(cong_tagh).shape)
# for c in range(53):
#    avg_cong.append(0)
#    max.append(0)
#    avg_cong.append(0)

# for c in range(53):
#   for i in range(81):
#     for j in range(60):
#         print(c)
#         print(cong_tagh[0][c][i][j])
#         if(max[c] < cong_tagh[c][i][j]):
#           max[c] =  cong_tagh[c][i][j]


# for c in range(54):
#   for i in range(81):
#     for j in range(60):
#         if(max_width < cong_tagh[c][i][j]):
#           avg_cong[c]+=cong_tagh[c][i][j]-max_width
          
# print(len(cong_tagh),avg_cong)
# 
  #  for j in i:
  #     print(j.shape)
      
# exit()
file_address="/home/saba/DL/time pred/"
# # # "/home/saba/DL/time pred"
# # # "/home/saba/DL/time pred/dataset/blif-res/stratixiv_arch.timing.xml"
# # # # "/home/saba/DL/time pred/dataset/res-other-titan/stratixiv_arch.timing.xml"
# # file_address ="/media/saba/Untitled/congest"
# f= file_address + "/gf6.txt"
# with open(f,'r') as data_file:
#       for line in data_file:

#           data = line.strip().split(',')
#           res=data
#           # print([data.float()])
#           # edge.append(np.asarray(data))
# res = [float(x) for x in res] 
# # # print(res)




# # print(len(gf1))
# # print("-------------")
# # print(len(res))
# # print(res)

# for i in range(len(gf1)):
#    for j in range(len(gf1[i])):
#       # print(gf1[i], res)
#       gf1[i][j] = gf1[i][j]/res[j]
# gf1 = torch.tensor(gf1,dtype=float) 
# # # res1=[]
# # # print("=====================================================")

# f1= file_address + "/inp6.txt"
# with open(f1,'r') as data_file:
#       for line in data_file:

#           data = line.strip().split(',')
#           res1=data
#           # print([float(data)])
# res1 = [float(x) for x in res1]
# # print(res1)

# for i in range(len(input_cnn)):
#   for j in range(len(input_cnn[i])):
#     #  print(input_cnn[i][j].max(),res1[j])
#      input_cnn[i][j] = input_cnn[i][j]/res1[j]

# print(gf1)
# print(input_cnn)
# exit()
# ######################################inp train norm##################################333333333333
# max_inp = [in_minx1,in_miny1,in_xmax1,in_ymax1,out_minx1,out_miny1,out_xmax1,out_ymax1,fin1,fout1,cong_tagh1]

max_inp = [in_minx1,in_miny1,in_xmax1,in_ymax1,fin1,fout1,cong_tagh1]

print(len(input_cnn))

# print(len(input_cnn[0]))
# print(len(input_cnn[0][0]))
# print( cong_tagh1)

for i in range(len(max_inp)):
   if max_inp[i] == 0 :
  #  and  max_inp[i] <= 1:
      max_inp[i] =1

# for i in range(len(input_cnn)):
#    for j in range(len(input_cnn[i])):
#     input_cnn[i][j] = input_cnn[i][j]/cong_tagh1






f = open(file_address + "/inp9.txt", "w")
for i in range(len(max_inp)):
    # for j in max_inp[i]:

    f.write(str(float(max_inp[i])))
    if(i != len(max_inp) - 1):
        f.write(",")
f.close()

for i in range(len(input_cnn)):
  for j in range(len(input_cnn[i])):
     print(input_cnn[i][j].max(),max_inp[j])
     input_cnn[i][j] = input_cnn[i][j]/max_inp[j]
# #####################################gf train norm#####################################33
print(gf1)
max_col = np.max(np.asarray(gf1), axis=0)

print(max_col)
print(torch.tensor(max_col).shape,torch.tensor(gf1).shape)
# # exit()
for i in range(len(max_col)):
   if max_col[i] == 0  :
  #  and  max_col[i] <= 1:
      print(max_col[i])
      max_col[i] =1
# # print(max_col)

f = open(file_address + "/gf9.txt", "w")
for i in range(len(max_col)):
    # for j in max_col[i]:

    f.write(str(float(max_col[i])))
    if(i != len(max_col) - 1):
        f.write(",")
# f.close()

for i in range(len(gf1)):
   for j in range(len(gf1[i])):
    # print(len(gf1[i]), len(max_col))
      gf1[i][j] = gf1[i][j]/max_col[j]
    
gf1 = torch.tensor(gf1,dtype=float) 
# exit()
# #####################################################################################################3333
# out=[]
# output=[]
# # out.append(list(output[0]))
# f = open(file_address + "/gf.txt", "r")

# for i in range(len(max_col)):
#     # for j in out[i]:
#     f.write(str(float(max_col[i])))
#     if(i != len(max_col) - 1):
#         f.write(",")
# f1 = open(file_address + "/inp.txt", "r")
# for i in range(len(max_inp)):
#     # for j in out[i]:
#     f1.write(str(float(max_inp[i])))
#     if(i != len(max_inp) - 1):
#         f1.write(",")
# # gf1 = gf1/gf1.max()
# # print(torch.isnan(input_cnn).sum())
# # print(torch.isnan(gf1).sum())
# # print(gf1[0])
# exit()
# max_col1 = np.max(np.asarray(gf1), axis=0)
# # print(max_col1)
# for i in range(len(max_col1)):
#    if max_col1[i] >= 0 and  max_col1[i] <= 1:
#       max_col1[i] =1
# print("----------------------------")
# print(max_col1)
# input_cnn = np.divide(np.asarray(input_cnn), max_col1) #

# input_cnn = torch.tensor(input_cnn,dtype=float) 

# for i in range(len(routability)):
#    if routability[i] == [0] or time_r[i] == 0:

#         time_r[i] = 3868194

# for i in range(len(routability)):
#    if routability[i] == [0]:
#       print(time_r[i], i)
#       if time_r[i] == 0:
#          time_r[i] = 3600000
                     
      # 
# exit()

# itrr=torch.tensor(itry)
# itr_max = itrr/itrr.max()

time_r = torch.tensor(time_r)
print("time_r")
for i in time_r:
   print(i)
print("------------+++++++++++----------------------")
time_max = time_r/time_r.max()
print(time_r.max())
# exit()
inputt=[]
# for i in range(len(input_cnn)):
#   input_cnn[i] = input_cnn[i].view(1,11,81,60)
  # inputt=torch.stack((input_cnn[i].view(1,11,81,60)),dim=0)
  # print(i.shape)
# inputt = torch.stack((input_cnn), dim=0)
# print(inputt.shape)

# print(time_max)
# for i in inputt:
#   print(i.shape)
# exit()
# print(time_r.max(),time_max)

time_rr = time_r
# x=0
# # # y = 0
# for i in range(len(routability)):
#   if routability[i] == [0]:
# #     x+=1
#     print(time_rr[i],i)
#     time_rr[i] = 3600000
# #                 #  1149237
# #                 #  2123118
#   # if time_rr[i] > 3600000:
#     # y+=1

# time_maxx = time_rr/3700000

# print("---------avvv--------")
# for ch in range(1,54):
#   for name in range(len(blif)):
#      print(av_mtx[ch][name])
#      gf1.ap
# ss=[]
# for i in av_mtx:
#    ss.append(i)
#    print(i)

# # print(torch.isnan(time_maxx).sum())
print("-----------------")

# print(torch.isnan(inputt).sum())
# exit()
out=[]
# # # # print(gf1.shape, inputt.shape)
model=CNN_croute() 

print("shapepe", gf1.shape)
# /blif_time7.pt
# '/oth-tit_time11.pt
# file_address ="/media/saba/Untitled/congest"
# file_address="/home/saba/DL/time pred/dataset/blif-res2/stratixiv_arch.timing.xml"
# model.load_state_dict(torch.load(file_address +'/blif_time19.pt'))
# params=[]
# for name, param in model.named_parameters():
# # for name, param in model.layer1.parameters():
#    print(name)
#    print(param.shape)

#   #  print(torch.sigmoid(param),'\n')
#   #  print(torch.sum(param))

#   #  params.append(param.view(-1))
#   #  params = torch.cat(params)

#   #  print(torch.sigmoid(param),'\n')
#   #  print("GGGGGG",torch.mean(torch.sigmoid(param)))

#   #  params_list = params.tolist()
#   #  break
# print("-----------------")
# print(len(param),param.shape)
# for i in param:
#     print(torch.mean(torch.sigmoid(i)))
#     # print(torch.sigmoid(i),'\n')

# sum_of_weights={el:0 for el in np.arange(11)}
# i = 0
# summarised_weight = 0
# for param in params_list:
#     sum_of_weights[i%11] += param
#     i += 1
# print(sum_of_weights)


# print(np.absolute(np.asarray(list(sum_of_weights.items()))))
# [[ 0.          2.15490737]
#  [ 1.          1.87546076]
#  [ 2.          1.15187944]
#  [ 3.          0.10264976]
#  [ 4.          1.13442085]
#  [ 5.          0.90985042]
#  [ 6.          0.69904608]
#  [ 7.          1.12858947]
#  [ 8.          0.24112399]
#  [ 9.          1.02664171]
#  [10.          1.55302067]]

# [0.0, 13.834518432617188]
# [1.0, 12.057506561279297]
# [2.0, 7.400966644287109]
# [3.0, 0.6602776646614075]
# [4.0, 7.2902703285217285]
# [5.0, 5.849956512451172]
# [6.0, 4.494727611541748]
# [7.0, 7.255417823791504]
# [8.0, 1.550149917602539]
# [9.0, 6.601776123046875]
# [10.0, 9.993911743164062]

# exit()
# model.eval()
# # # # # # # model.train()
# input_cnn = torch.stack((input_cnn), dim=0)
# # # # input_cnn = torch.tensor(input_cnn, dtype=float)
# # # # # # # print("inpppppp",input_cnn.shape)
# output = model(input_cnn.float(),gf1.float())

# # # for i in list(output):
# # #    print(i)
# # # # out.append(list(output[0]))
# # # # f = open(file_address + "/out.txt", "w")
# # # # for i in range(len(out)):
# # # #     for j in out[i]:

# # # #         f.write(str(float(j)))
# # # #     if(i != len(out) - 1):
# # # #         f.write(",")
# # # # f.close()


# outt=[]
# for i in output:
# #    outt.append(float(i[0])*3908556)
# #    print(float(i[0])*3908556)
#    outt.append(float(i[0])*3868194)
#    print(float(i[0])*3868194)
# timee=[]
# print("--------------")
# for i in ss:
#    print(float(i))
# # print("--------------")
# for i in time_r:
#    timee.append(float(i))
#    print(float(i))
# import scipy.stats as stats

# cong = np.array(ss)
# pred = np.array(outt)
# timee = np.array(timee)
# sort= np.argsort(timee)


# print("=====================")
# for i in timee[sort]:
#    print(i)
# print("=====================")
# for i in pred[sort]:
#    print(i)
# print("=====================")
# for i in cong[sort]:
  #  print(i)
# wl=[690.3, 798.705, 798.705, 796.838, 794.205, 757.334, 752.121, 799.518, 798.072,797.512, 773.163, 768.21, 744.496, 747.149, 799.303, 796.894, 793.874, 774.486, 780.143, 736.99, 750.492, 729.438, 726.792, 798.072, 797.5, 669.127, 680.62, 797.573, 795.583, 785.074, 769.547, 797.573, 795.718, 797.573, 791.701, 702.742, 707.002, 797.573, 794.629, 723.561, 739.473, 797.573, 794.629, 797.573, 794.629, 724.358, 738.115, 797.573, 796.792, 794.629]
# wl = np.array(wl)
# # print(wl)
# # # print(wl[sort])
# tau, p_value = stats.kendalltau(timee[sort], pred[sort])
# print(tau)
# tau, p_value = stats.kendalltau(timee[sort], cong[sort])
# print(tau)
# tau, p_value = stats.kendalltau(timee[sort], wl[sort])
# print(tau)
# print(pred[sort],timee[sort])

# plt.plot(pred[sort],timee[sort], 'o')
# plt.show()


# print("TIME ", time.time() - start)
# exit()
# print(len(fpga_size))
# exit()
import random

dataas=[]
for i in range(len(fpga_size)):
   dataas.append(i)

random.shuffle(dataas)
train_indices = dataas[int(len(fpga_size)*0.8):]
test_indices = dataas[:int(len(fpga_size)*0.8)]

# shuftr=[]
# shufts=[]
# print(input_cnn[0][0])
# for i in range(334):
#    shuftr.append(i+1)

# for i in range(84):
#    shufts.append(i+1)
# random.shuffle(shuftr)
# random.shuffle(shufts)
# print(shuftr)
# print(shufts)
# # exit()
# print(input_cnn[0])
# input_cnn[0] = input_cnn[1]
# print(input_cnn[0])
# shufts=[63, 61, 37, 14, 5, 30, 50, 24, 22, 71, 324, 315, 318, 54, 130, 113, 193, 47, 118, 132, 16, 38, 42, 3, 55, 39, 48, 72, 8,  36, 23, 67, 41, 15, 11, 33, 52, 43, 75, 79, 56, 6,  47, 2, 65, 77, 13, 76, 82, 7, 34, 27, 74, 35, 22, 70, 59, 78, 68, 83, 44, 53, 314, 10, 177, 191, 88, 176, 187, 140, 204, 223, 17, 194,110, 87, 236, 136, 25, 86, 317, 201, 156, 51]

# # shufts=[13, 71, 33, 50, 35, 43, 65, 31, 85, 36, 5, 27, 78, 77, 32, 66, 10, 2, 88, 74, 89, 79, 82, 3, 8, 15, 20, 25, 76, 11, 84, 86, 18, 56, 54, 23, 28, 38, 83, 16, 47, 73, 72, 62, 29, 69, 53, 34, 63, 75, 45, 42, 58, 37, 24, 19, 9, 92, 67, 48, 4, 30, 64, 7, 1, 90, 87, 51, 91, 61, 44, 60, 70, 6, 46, 52, 21, 40, 80, 39, 59, 57, 17, 49, 41, 12, 22, 68, 26, 81, 14, 55]
# # input_cnn1[]
# for i in range(len(train_indices)):
# #   #  364
#     input_cnn[int(len(fpga_size)*0.8):][i]=input_cnn[train_indices[i]] 
#     gf1[int(len(fpga_size)*0.8):][i]=gf1[train_indices[i]] 
#     time_max[int(len(fpga_size)*0.8):][i]=time_max[train_indices[i]] 
# # # shuftr=[275, 72, 216, 137, 33, 300, 273, 12, 122, 175, 245, 267, 217, 198,  225, 43, 202, 305, 231, 139, 237, 249, 125, 56, 133, 180, 271, 50, 179, 309, 68, 35, 188, 243, 306, 91, 304, 114, 174, 54, 58, 26, 40, 18, 12, 73, 20, 80, 66, 9, 49, 84, 57, 17, 31, 21, 32, 28, 10, 25, 51, 19, 71, 295, 120, 157, 34, 27, 197, 263, 312, 41, 40, 109, 97, 81, 332, 107, 227, 208, 95, 287, 185, 65, 143, 261, 265, 105, 286, 37, 282, 276, 247,  159, 53, 235, 158, 333, 164, 248, 292, 162, 184, 101, 233, 116, 289, 84, 146, 294, 281, 266, 73, 42, 327, 134, 69, 102, 78, 61, 211, 48, 172, 232, 181, 131, 60, 321, 241, 238, 183, 104, 264, 326, 77, 220, 145, 39, 173, 36, 268, 303, 182, 38, 58, 90, 3, 128, 147, 246, 279, 302, 64, 334, 108, 28, 221, 154, 213, 63, 98, 93, 70, 96, 163, 251, 9, 165, 62, 121, 230, 229, 205, 215, 297, 288, 100, 253, 160, 280, 117, 218, 186, 310, 30, 259, 299, 325, 329, 126, 66, 57, 13, 319, 195, 296, 141, 26, 99, 242, 7, 151, 226, 209, 272, 44, 252, 55, 224, 277, 20, 166, 239, 274, 124, 67, 250, 49, 200, 207, 283, 284, 94, 135, 29, 313, 328, 149, 127, 16, 24, 308, 32, 150, 23, 2, 257, 19, 123, 153, 75, 5, 291, 298, 222, 82, 290, 196, 228, 144, 316, 171, 170, 331, 138, 115, 206, 168, 83, 4, 80, 219, 111, 45, 167, 46, 244, 212, 190, 6, 285, 8, 89, 103, 21, 15, 76, 270, 210, 262, 256, 14, 59, 152, 79, 178, 189, 92, 255, 148, 85, 18, 112, 74, 320, 192, 278, 106, 293, 161, 269, 169, 119, 240, 129, 322, 258, 11, 31, 142, 307, 330, 214, 155, 311, 52, 301, 1, 260, 199, 234, 323, 254, 203, 69, 62, 29, 46, 1, 60, 64, 81, 45, 4]
# # # # shuftr=[289, 73, 128, 328, 211, 167, 190, 337, 80, 202, 325, 327, 124, 270, 60, 53, 91, 276, 208, 88, 94, 5, 230, 302, 17, 92, 20, 232, 354, 247, 7, 27, 179, 198, 294, 330, 72, 30, 9, 74, 75, 349, 362, 182, 68, 286, 217, 364, 257, 221, 226, 238, 97, 231, 332, 245, 52, 207, 122, 333, 272, 258, 154, 275, 25, 77, 240, 113, 65, 306, 350, 218, 214, 85, 135, 196, 61, 18, 296, 254, 309, 100, 323, 290, 209, 203, 316, 105, 58, 8, 141, 132, 28, 334, 287, 223, 40, 224, 11, 293, 89, 250, 78, 14, 204, 338, 308, 244, 13, 251, 262, 140, 22, 95, 317, 303, 237, 143, 49, 288, 266, 24, 31, 345, 2, 188, 109, 261, 131, 213, 129, 15, 219, 50, 26, 48, 157, 117, 299, 169, 321, 282, 137, 357, 118, 242, 121, 246, 144, 268, 310, 96, 300, 120, 280, 322, 47, 351, 228, 234, 130, 178, 194, 352, 156, 340, 319, 263, 318, 1, 12, 4, 189, 71, 149, 127, 278, 326, 267, 39, 311, 164, 342, 329, 216, 235, 301, 283, 42, 199, 348, 277, 298, 3, 160, 313, 259, 106, 76, 41, 107, 147, 33, 177, 360, 180, 152, 110, 271, 255, 355, 312, 220, 84, 63, 108, 37, 205, 23, 356, 256, 186, 46, 307, 273, 138, 358, 264, 346, 170, 265, 236, 158, 324, 38, 361, 21, 239, 305, 279, 173, 101, 284, 176, 134, 200, 193, 159, 104, 35, 183, 136, 146, 248, 99, 139, 314, 166, 197, 32, 171, 304, 187, 54, 274, 227, 252, 260, 125, 93, 292, 87, 233, 331, 281, 336, 56, 174, 44, 269, 126, 103, 335, 98, 363, 168, 62, 295, 285, 148, 222, 81, 344, 215, 6, 206, 359, 165, 347, 43, 69, 153, 163, 66, 83, 79, 172, 201, 116, 36, 145, 229, 133, 119, 210, 67, 353, 184, 191, 34, 29, 57, 185, 45, 51, 249, 112, 151, 315, 195, 181, 212, 82, 16, 155, 115, 59, 241, 64, 86, 297, 291, 123, 175, 114, 320, 339, 162, 192, 243, 102, 90, 161, 111, 225, 19, 341, 142, 253, 70, 10, 150, 343, 55]

# for i in range(len(test_indices)):
#     input_cnn[:int(len(fpga_size)*0.8)][i]=input_cnn[test_indices[i]] 
#     gf1[:int(len(fpga_size)*0.8)][i]=gf1[test_indices[i]] 
#     time_max[:int(len(fpga_size)*0.8)][i]=time_max[test_indices[i]] 




# exit()
# tr = shuftr[:196]
# tst = shufts[196:]
# shuf = input_cnn[tr]
# input_cnn_tr= shuftr
# gf_tr =  shuftr
# time_max_tr =  shuftr
# print(shufts)
# shufts=[13, 71, 33, 50, 35, 43, 65, 31, 85, 36, 5, 27, 78, 77, 32, 66, 10, 2, 88, 74, 89, 79, 82, 3, 8, 15, 20, 25, 76, 11, 84, 86, 18, 56, 54, 23, 28, 38, 83, 16, 47, 73, 72, 62, 29, 69, 53, 34, 63, 75, 45, 42, 58, 37, 24, 19, 9, 92, 67, 48, 4, 30, 64, 7, 1, 90, 87, 51, 91, 61, 44, 60, 70, 6, 46, 52, 21, 40, 80, 39, 59, 57, 17, 49, 41, 12, 22, 68, 26, 81, 14, 55]

# # exit()
# print(gf1)
# print(len(gf1))
# print(len(input_cnn))
# # exit()
# shuftr=[289, 73, 128, 328, 211, 167, 190, 337, 80, 202, 325, 327, 124, 270, 60, 53, 91, 276, 208, 88, 94, 5, 230, 302, 17, 92, 20, 232, 354, 247, 7, 27, 179, 198, 294, 330, 72, 30, 9, 74, 75, 349, 362, 182, 68, 286, 217, 364, 257, 221, 226, 238, 97, 231, 332, 245, 52, 207, 122, 333, 272, 258, 154, 275, 25, 77, 240, 113, 65, 306, 350, 218, 214, 85, 135, 196, 61, 18, 296, 254, 309, 100, 323, 290, 209, 203, 316, 105, 58, 8, 141, 132, 28, 334, 287, 223, 40, 224, 11, 293, 89, 250, 78, 14, 204, 338, 308, 244, 13, 251, 262, 140, 22, 95, 317, 303, 237, 143, 49, 288, 266, 24, 31, 345, 2, 188, 109, 261, 131, 213, 129, 15, 219, 50, 26, 48, 157, 117, 299, 169, 321, 282, 137, 357, 118, 242, 121, 246, 144, 268, 310, 96, 300, 120, 280, 322, 47, 351, 228, 234, 130, 178, 194, 352, 156, 340, 319, 263, 318, 1, 12, 4, 189, 71, 149, 127, 278, 326, 267, 39, 311, 164, 342, 329, 216, 235, 301, 283, 42, 199, 348, 277, 298, 3, 160, 313, 259, 106, 76, 41, 107, 147, 33, 177, 360, 180, 152, 110, 271, 255, 355, 312, 220, 84, 63, 108, 37, 205, 23, 356, 256, 186, 46, 307, 273, 138, 358, 264, 346, 170, 265, 236, 158, 324, 38, 361, 21, 239, 305, 279, 173, 101, 284, 176, 134, 200, 193, 159, 104, 35, 183, 136, 146, 248, 99, 139, 314, 166, 197, 32, 171, 304, 187, 54, 274, 227, 252, 260, 125, 93, 292, 87, 233, 331, 281, 336, 56, 174, 44, 269, 126, 103, 335, 98, 363, 168, 62, 295, 285, 148, 222, 81, 344, 215, 6, 206, 359, 165, 347, 43, 69, 153, 163, 66, 83, 79, 172, 201, 116, 36, 145, 229, 133, 119, 210, 67, 353, 184, 191, 34, 29, 57, 185, 45, 51, 249, 112, 151, 315, 195, 181, 212, 82, 16, 155, 115, 59, 241, 64, 86, 297, 291, 123, 175, 114, 320, 339, 162, 192, 243, 102, 90, 161, 111, 225, 19, 341, 142, 253, 70, 10, 150, 343, 55]
# print(gf_tr,len(gf_tr))
# for i in range(len(shuftr)):
#    print(shuftr[i])
#    input_cnn_tr[i] = input_cnn[shuftr[i]]
#    shuftr=[289, 73, 128, 328, 211, 167, 190, 337, 80, 202, 325, 327, 124, 270, 60, 53, 91, 276, 208, 88, 94, 5, 230, 302, 17, 92, 20, 232, 354, 247, 7, 27, 179, 198, 294, 330, 72, 30, 9, 74, 75, 349, 362, 182, 68, 286, 217, 364, 257, 221, 226, 238, 97, 231, 332, 245, 52, 207, 122, 333, 272, 258, 154, 275, 25, 77, 240, 113, 65, 306, 350, 218, 214, 85, 135, 196, 61, 18, 296, 254, 309, 100, 323, 290, 209, 203, 316, 105, 58, 8, 141, 132, 28, 334, 287, 223, 40, 224, 11, 293, 89, 250, 78, 14, 204, 338, 308, 244, 13, 251, 262, 140, 22, 95, 317, 303, 237, 143, 49, 288, 266, 24, 31, 345, 2, 188, 109, 261, 131, 213, 129, 15, 219, 50, 26, 48, 157, 117, 299, 169, 321, 282, 137, 357, 118, 242, 121, 246, 144, 268, 310, 96, 300, 120, 280, 322, 47, 351, 228, 234, 130, 178, 194, 352, 156, 340, 319, 263, 318, 1, 12, 4, 189, 71, 149, 127, 278, 326, 267, 39, 311, 164, 342, 329, 216, 235, 301, 283, 42, 199, 348, 277, 298, 3, 160, 313, 259, 106, 76, 41, 107, 147, 33, 177, 360, 180, 152, 110, 271, 255, 355, 312, 220, 84, 63, 108, 37, 205, 23, 356, 256, 186, 46, 307, 273, 138, 358, 264, 346, 170, 265, 236, 158, 324, 38, 361, 21, 239, 305, 279, 173, 101, 284, 176, 134, 200, 193, 159, 104, 35, 183, 136, 146, 248, 99, 139, 314, 166, 197, 32, 171, 304, 187, 54, 274, 227, 252, 260, 125, 93, 292, 87, 233, 331, 281, 336, 56, 174, 44, 269, 126, 103, 335, 98, 363, 168, 62, 295, 285, 148, 222, 81, 344, 215, 6, 206, 359, 165, 347, 43, 69, 153, 163, 66, 83, 79, 172, 201, 116, 36, 145, 229, 133, 119, 210, 67, 353, 184, 191, 34, 29, 57, 185, 45, 51, 249, 112, 151, 315, 195, 181, 212, 82, 16, 155, 115, 59, 241, 64, 86, 297, 291, 123, 175, 114, 320, 339, 162, 192, 243, 102, 90, 161, 111, 225, 19, 341, 142, 253, 70, 10, 150, 343, 55]
#    print(shuftr[i])
#   #  exit()
#    print(gf1[shuftr[i]])
#    gf_tr[i]=gf1[shuftr[i]]
#    shuftr=[289, 73, 128, 328, 211, 167, 190, 337, 80, 202, 325, 327, 124, 270, 60, 53, 91, 276, 208, 88, 94, 5, 230, 302, 17, 92, 20, 232, 354, 247, 7, 27, 179, 198, 294, 330, 72, 30, 9, 74, 75, 349, 362, 182, 68, 286, 217, 364, 257, 221, 226, 238, 97, 231, 332, 245, 52, 207, 122, 333, 272, 258, 154, 275, 25, 77, 240, 113, 65, 306, 350, 218, 214, 85, 135, 196, 61, 18, 296, 254, 309, 100, 323, 290, 209, 203, 316, 105, 58, 8, 141, 132, 28, 334, 287, 223, 40, 224, 11, 293, 89, 250, 78, 14, 204, 338, 308, 244, 13, 251, 262, 140, 22, 95, 317, 303, 237, 143, 49, 288, 266, 24, 31, 345, 2, 188, 109, 261, 131, 213, 129, 15, 219, 50, 26, 48, 157, 117, 299, 169, 321, 282, 137, 357, 118, 242, 121, 246, 144, 268, 310, 96, 300, 120, 280, 322, 47, 351, 228, 234, 130, 178, 194, 352, 156, 340, 319, 263, 318, 1, 12, 4, 189, 71, 149, 127, 278, 326, 267, 39, 311, 164, 342, 329, 216, 235, 301, 283, 42, 199, 348, 277, 298, 3, 160, 313, 259, 106, 76, 41, 107, 147, 33, 177, 360, 180, 152, 110, 271, 255, 355, 312, 220, 84, 63, 108, 37, 205, 23, 356, 256, 186, 46, 307, 273, 138, 358, 264, 346, 170, 265, 236, 158, 324, 38, 361, 21, 239, 305, 279, 173, 101, 284, 176, 134, 200, 193, 159, 104, 35, 183, 136, 146, 248, 99, 139, 314, 166, 197, 32, 171, 304, 187, 54, 274, 227, 252, 260, 125, 93, 292, 87, 233, 331, 281, 336, 56, 174, 44, 269, 126, 103, 335, 98, 363, 168, 62, 295, 285, 148, 222, 81, 344, 215, 6, 206, 359, 165, 347, 43, 69, 153, 163, 66, 83, 79, 172, 201, 116, 36, 145, 229, 133, 119, 210, 67, 353, 184, 191, 34, 29, 57, 185, 45, 51, 249, 112, 151, 315, 195, 181, 212, 82, 16, 155, 115, 59, 241, 64, 86, 297, 291, 123, 175, 114, 320, 339, 162, 192, 243, 102, 90, 161, 111, 225, 19, 341, 142, 253, 70, 10, 150, 343, 55]
   
#    time_max_tr[i]=time_max[shuftr[i]]
#    shuftr=[289, 73, 128, 328, 211, 167, 190, 337, 80, 202, 325, 327, 124, 270, 60, 53, 91, 276, 208, 88, 94, 5, 230, 302, 17, 92, 20, 232, 354, 247, 7, 27, 179, 198, 294, 330, 72, 30, 9, 74, 75, 349, 362, 182, 68, 286, 217, 364, 257, 221, 226, 238, 97, 231, 332, 245, 52, 207, 122, 333, 272, 258, 154, 275, 25, 77, 240, 113, 65, 306, 350, 218, 214, 85, 135, 196, 61, 18, 296, 254, 309, 100, 323, 290, 209, 203, 316, 105, 58, 8, 141, 132, 28, 334, 287, 223, 40, 224, 11, 293, 89, 250, 78, 14, 204, 338, 308, 244, 13, 251, 262, 140, 22, 95, 317, 303, 237, 143, 49, 288, 266, 24, 31, 345, 2, 188, 109, 261, 131, 213, 129, 15, 219, 50, 26, 48, 157, 117, 299, 169, 321, 282, 137, 357, 118, 242, 121, 246, 144, 268, 310, 96, 300, 120, 280, 322, 47, 351, 228, 234, 130, 178, 194, 352, 156, 340, 319, 263, 318, 1, 12, 4, 189, 71, 149, 127, 278, 326, 267, 39, 311, 164, 342, 329, 216, 235, 301, 283, 42, 199, 348, 277, 298, 3, 160, 313, 259, 106, 76, 41, 107, 147, 33, 177, 360, 180, 152, 110, 271, 255, 355, 312, 220, 84, 63, 108, 37, 205, 23, 356, 256, 186, 46, 307, 273, 138, 358, 264, 346, 170, 265, 236, 158, 324, 38, 361, 21, 239, 305, 279, 173, 101, 284, 176, 134, 200, 193, 159, 104, 35, 183, 136, 146, 248, 99, 139, 314, 166, 197, 32, 171, 304, 187, 54, 274, 227, 252, 260, 125, 93, 292, 87, 233, 331, 281, 336, 56, 174, 44, 269, 126, 103, 335, 98, 363, 168, 62, 295, 285, 148, 222, 81, 344, 215, 6, 206, 359, 165, 347, 43, 69, 153, 163, 66, 83, 79, 172, 201, 116, 36, 145, 229, 133, 119, 210, 67, 353, 184, 191, 34, 29, 57, 185, 45, 51, 249, 112, 151, 315, 195, 181, 212, 82, 16, 155, 115, 59, 241, 64, 86, 297, 291, 123, 175, 114, 320, 339, 162, 192, 243, 102, 90, 161, 111, 225, 19, 341, 142, 253, 70, 10, 150, 343, 55]
   

# input_cnn_ts=shufts
# gf_ts=shufts
# time_max_ts=shufts

# for i in range(len(shufts)):
#    shufts=[13, 71, 33, 50, 35, 43, 65, 31, 85, 36, 5, 27, 78, 77, 32, 66, 10, 2, 88, 74, 89, 79, 82, 3, 8, 15, 20, 25, 76, 11, 84, 86, 18, 56, 54, 23, 28, 38, 83, 16, 47, 73, 72, 62, 29, 69, 53, 34, 63, 75, 45, 42, 58, 37, 24, 19, 9, 92, 67, 48, 4, 30, 64, 7, 1, 90, 87, 51, 91, 61, 44, 60, 70, 6, 46, 52, 21, 40, 80, 39, 59, 57, 17, 49, 41, 12, 22, 68, 26, 81, 14, 55]

#    input_cnn_ts[i] = input_cnn[shufts[i]]
#    shufts=[13, 71, 33, 50, 35, 43, 65, 31, 85, 36, 5, 27, 78, 77, 32, 66, 10, 2, 88, 74, 89, 79, 82, 3, 8, 15, 20, 25, 76, 11, 84, 86, 18, 56, 54, 23, 28, 38, 83, 16, 47, 73, 72, 62, 29, 69, 53, 34, 63, 75, 45, 42, 58, 37, 24, 19, 9, 92, 67, 48, 4, 30, 64, 7, 1, 90, 87, 51, 91, 61, 44, 60, 70, 6, 46, 52, 21, 40, 80, 39, 59, 57, 17, 49, 41, 12, 22, 68, 26, 81, 14, 55]

#    gf_ts[i]=gf1[shufts[i]]
#    shufts=[13, 71, 33, 50, 35, 43, 65, 31, 85, 36, 5, 27, 78, 77, 32, 66, 10, 2, 88, 74, 89, 79, 82, 3, 8, 15, 20, 25, 76, 11, 84, 86, 18, 56, 54, 23, 28, 38, 83, 16, 47, 73, 72, 62, 29, 69, 53, 34, 63, 75, 45, 42, 58, 37, 24, 19, 9, 92, 67, 48, 4, 30, 64, 7, 1, 90, 87, 51, 91, 61, 44, 60, 70, 6, 46, 52, 21, 40, 80, 39, 59, 57, 17, 49, 41, 12, 22, 68, 26, 81, 14, 55]

#    time_max_ts[i]=time_max[shufts[i]]
#    shufts=[13, 71, 33, 50, 35, 43, 65, 31, 85, 36, 5, 27, 78, 77, 32, 66, 10, 2, 88, 74, 89, 79, 82, 3, 8, 15, 20, 25, 76, 11, 84, 86, 18, 56, 54, 23, 28, 38, 83, 16, 47, 73, 72, 62, 29, 69, 53, 34, 63, 75, 45, 42, 58, 37, 24, 19, 9, 92, 67, 48, 4, 30, 64, 7, 1, 90, 87, 51, 91, 61, 44, 60, 70, 6, 46, 52, 21, 40, 80, 39, 59, 57, 17, 49, 41, 12, 22, 68, 26, 81, 14, 55]

# print(input_cnn_tr)
# exit()
# train_dataset = CustomImageDataset(input_cnn,time_max_tr,gf_tr)
# test_dataset = CustomImageDataset(input_cnn_ts,time_max_ts,gf_ts)
sz = len(input_cnn)
prop=8/10
# train_dataset = CustomImageDataset(input_cnn[:1],time_r[:1])
train_dataset = CustomImageDataset(input_cnn[:int(sz*prop)],time_max[:int(sz*prop)],gf1[:int(sz*prop)])
test_dataset = CustomImageDataset(input_cnn[int(sz*prop):],time_max[int(sz*prop):],gf1[int(sz*prop):])
print("len",test_dataset.__len__(), train_dataset.__len__())
# print(input_cnn[:int(sz*prop)][0])
# exit()
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# model=CNN_croute(11,1)
#
val_accu = []
train_accu = []

val_loss = []
train_los = []
# Number of epochs to train the model
learning_rate=0.001
# Learning rate for the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# Loss function to minimize #
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Optimizer to update the model parameters

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Adam optimizer #
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# criterion = torch.nn.BCELoss()
# print("losssssssss",test1(model,train_dataloader,criterion,time_r.max()))
# print(test1(model,test_dataloader,criterion,time_r.max()))
for epoch in range(1, 70):
  print("EPOCH ",epoch)
  train1(model,train_dataloader,criterion,optimizer)
  # train1(train_indices, batch, gf,GAT,criterion,optimizer,scheduler)
  print("-----------------train--------------------------")
  train_loss,acttr = test1(model,train_dataloader,criterion,time_r.max())

  print("++++++++++++++++++test++++++++++++++++++++++++")
  start = time.time()

  test_loss, acts = test1(model,test_dataloader,criterion,time_r.max())

  train_los.append(train_loss)
  val_loss.append(test_loss)

  train_accu.append(acttr * 100)
  val_accu.append(acts * 100)
  
  print("TIME ", time.time() - start)
  print("Train",train_loss)
  print("Test",test_loss)
  torch.save(model.state_dict(), file_address +'/blif_time21.pt')

torch.save(model.state_dict(), file_address +'/blif_time21.pt')
#  ,acc,p,rec,spec )
  # print(max(time_r_G))
# canc=list(model.cancelout.parameters())[0].detach().numpy()
  #print(max(time_r_G))
import matplotlib.pyplot as plt
#plt.figure(figsize=(10,5))
#plt.title("Training and Validation Loss")
#plt.ylim(0, 100)
#dtst_los = [fl.item() for fl in val_loss ]
#plt.plot(dtst_los,label="val")


#dtrn_los = [fl.item() for fl in train_los ]
#plt.plot(dtrn_los,label="train")
#plt.xlabel("iterations")
#plt.ylabel("acc")
#plt.legend()
#plt.show()


#plt.figure(figsize=(10,5))
#plt.title("Training and Validation Loss")
#plt.ylim(0, 100)

#dtst_ac = [fl for fl in val_accu ]
#plt.plot(dtst_ac,label="val")

#dttrn_ac = [fl for fl in train_accu ]
#plt.plot(dttrn_ac,label="train")
#plt.xlabel("iterations")
#plt.ylabel("acc")
#plt.legend()
#plt.show()


# 
# def feature_extraction(X, y, f_number):

# feature_weights = {}
# # Gradient Boosting FS
# # forest = GradientBoostingClassifier(n_estimators=50,random_state=0)
# # forest.fit(X, y)
# # gb_importances = forest.feature_importances_
# f_number=11
# # # NN FS
# cancelout_weights_importance = canc
# print("CancelOut weights after the activation function:")
# print(torch.sigmoid(torch.tensor(cancelout_weights_importance)),'\n')
# # selecting first 5 features
# feature_weights['nn'] = cancelout_weights_importance.argsort()[::-1]
# # feature_weights['gb'] = gb_importances.argsort()[::-1]
# nn_fi = cancelout_weights_importance.argsort()[-f_number:][::-1]
# # gb_fi = gb_importances.argsort()[-f_number:][::-1]

# print('Features selected using ANN with CancelOut', sorted(nn_fi))
# # print('Features selected using GB ',sorted(gb_fi))

# # print(f'CV score from all features: {CV_test( X, y)}')
# # print(f'CV score GB FS: {CV_test(X[:,gb_fi], y)}')
# # print(f'CV score NN FS: {CV_test(X[:,nn_fi], y)}')

# # return feature_weights
# fw = feature_weights



# CORRECT:  0.9566666666666667
# ++++++++++++++++++test++++++++++++++++++++++++
# OUTPUTS:  tensor([[2.2233e-03],
#         [6.4319e-06],
#         [6.4319e-06],
#         [2.7334e-06],
#         [1.0115e-01],
#         [7.4469e-02],
#         [4.1205e-02],
#         [7.2896e-02],
#         [8.5802e-06],
#         [6.7050e-02],
#         [2.6292e-04],
#         [3.4914e-02],
#         [6.7050e-02],
#         [9.8077e-01],
#         [3.4914e-02],
#         [2.8190e-02]], grad_fn=<ViewBackward0>)
# ---------------------
# tensor([[8.1821e+03],
#         [2.3670e+01],
#         [2.3670e+01],
#         [1.0059e+01],
#         [3.7226e+05],
#         [2.7406e+05],
#         [1.5164e+05],
#         [2.6827e+05],
#         [3.1576e+01],
#         [2.4675e+05],
#         [9.6759e+02],
#         [1.2849e+05],
#         [2.4675e+05],
#         [3.6094e+06],
#         [1.2849e+05],
#         [1.0374e+05]], grad_fn=<MulBackward0>) tensor([ 7,  1, 21, 41, 42, 66,  0, 74, 53, 26, 13, 20, 22,  6, 24, 56])
# LABEL:  tensor([[1.2418e-04],
#         [2.0543e-04],
#         [2.0217e-04],
#         [2.1684e-04],
#         [1.0028e-01],
#         [4.7422e-02],
#         [3.5124e-02],
#         [5.9039e-02],
#         [2.1874e-04],
#         [6.7458e-02],
#         [3.4238e-05],
#         [4.6517e-02],
#         [6.8387e-02],
#         [4.0231e-01],
#         [4.7476e-02],
#         [9.9484e-02]])
# ---------------------
# tensor([[4.5700e+02],
#         [7.5600e+02],
#         [7.4400e+02],
#         [7.9800e+02],
#         [3.6905e+05],
#         [1.7452e+05],
#         [1.2926e+05],
#         [2.1727e+05],
#         [8.0500e+02],
#         [2.4825e+05],
#         [1.2600e+02],
#         [1.7119e+05],
#         [2.5167e+05],
#         [1.4805e+06],
#         [1.7472e+05],
#         [3.6611e+05]])
# ACC:  tensor(1)
# ----------------------------------
# OUTPUTS:  tensor([[3.9707e-05],
#         [1.0115e-01],
#         [3.5057e-05],
#         [4.5422e-05],
#         [3.4375e-06],
#         [3.5192e-02],
#         [3.2709e-02],
#         [2.9621e-02],
#         [4.1442e-05],
#         [4.1512e-05],
#         [6.4319e-06],
#         [1.3268e-03],
#         [1.0115e-01],
#         [8.6612e-02],
#         [1.0000e+00],
#         [3.5057e-05]], grad_fn=<ViewBackward0>)
# ---------------------
# tensor([[1.4613e+02],
#         [3.7226e+05],
#         [1.2902e+02],
#         [1.6716e+02],
#         [1.2650e+01],
#         [1.2951e+05],
#         [1.2037e+05],
#         [1.0901e+05],
#         [1.5251e+02],
#         [1.5277e+02],
#         [2.3670e+01],
#         [4.8828e+03],
#         [3.7226e+05],
#         [3.1874e+05],
#         [3.6801e+06],
#         [1.2902e+02]], grad_fn=<MulBackward0>) tensor([51, 46, 39, 15, 33, 64, 36, 44, 59, 11, 25,  5, 50, 38,  9, 43])
# LABEL:  tensor([[1.6766e-04],
#         [9.9689e-02],
#         [1.7934e-04],
#         [1.6195e-04],
#         [2.3287e-04],
#         [4.8653e-02],
#         [3.6172e-02],
#         [3.6064e-02],
#         [1.7255e-04],
#         [2.1249e-04],
#         [2.0461e-04],
#         [1.7445e-04],
#         [9.9472e-02],
#         [1.2839e-01],
#         [9.9632e-01],
#         [1.9809e-04]])
# ---------------------
# tensor([[6.1700e+02],
#         [3.6687e+05],
#         [6.6000e+02],
#         [5.9600e+02],
#         [8.5700e+02],
#         [1.7905e+05],
#         [1.3312e+05],
#         [1.3272e+05],
#         [6.3500e+02],
#         [7.8200e+02],
#         [7.5300e+02],
#         [6.4200e+02],
#         [3.6607e+05],
#         [4.7248e+05],
#         [3.6666e+06],
#         [7.2900e+02]])
# ACC:  tensor(0)
# ----------------------------------
# OUTPUTS:  tensor([[3.4375e-06],
#         [8.0629e-04],
#         [2.8190e-02],
#         [3.4375e-06],
#         [2.7334e-06],
#         [3.7729e-02],
#         [4.1205e-02],
#         [8.6612e-02],
#         [2.9621e-02],
#         [4.1442e-05],
#         [3.7773e-04],
#         [6.4319e-06],
#         [5.2543e-05],
#         [9.3951e-01],
#         [4.5422e-05],
#         [8.5802e-06]], grad_fn=<ViewBackward0>)
# ---------------------
# tensor([[1.2650e+01],
#         [2.9673e+03],
#         [1.0374e+05],
#         [1.2650e+01],
#         [1.0059e+01],
#         [1.3885e+05],
#         [1.5164e+05],
#         [3.1874e+05],
#         [1.0901e+05],
#         [1.5251e+02],
#         [1.3901e+03],
#         [2.3670e+01],
#         [1.9337e+02],
#         [3.4575e+06],
#         [1.6716e+02],
#         [3.1576e+01]], grad_fn=<MulBackward0>) tensor([37, 14, 52, 29, 49, 68, 16, 34, 48, 63,  4, 17, 35,  8, 19, 65])
# LABEL:  tensor([[2.2635e-04],
#         [2.7616e-03],
#         [9.5743e-02],
#         [2.3314e-04],
#         [2.1276e-04],
#         [5.3260e-02],
#         [3.2386e-02],
#         [1.3300e-01],
#         [3.6226e-02],
#         [1.7282e-04],
#         [1.1358e-04],
#         [2.0135e-04],
#         [1.7336e-04],
#         [9.9414e-01],
#         [1.6766e-04],
#         [2.1657e-04]])
# ---------------------
# tensor([[8.3300e+02],
#         [1.0163e+04],
#         [3.5235e+05],
#         [8.5800e+02],
#         [7.8300e+02],
#         [1.9600e+05],
#         [1.1918e+05],
#         [4.8946e+05],
#         [1.3332e+05],
#         [6.3600e+02],
#         [4.1800e+02],
#         [7.4100e+02],
#         [6.3800e+02],
#         [3.6585e+06],
#         [6.1700e+02],
#         [7.9700e+02]])
# ACC:  tensor(0)
# ----------------------------------
# OUTPUTS:  tensor([[7.2896e-02],
#         [3.0739e-04],
#         [6.7050e-02],
#         [8.6612e-02],
#         [8.5802e-06],
#         [2.7334e-06],
#         [7.2185e-02],
#         [3.7729e-02],
#         [7.2185e-02],
#         [3.5192e-02],
#         [3.2709e-02],
#         [5.2543e-05],
#         [7.0965e-03],
#         [3.5057e-05],
#         [2.7334e-06],
#         [2.7334e-06]], grad_fn=<ViewBackward0>)
# ---------------------
# tensor([[2.6827e+05],
#         [1.1312e+03],
#         [2.4675e+05],
#         [3.1874e+05],
#         [3.1576e+01],
#         [1.0059e+01],
#         [2.6565e+05],
#         [1.3885e+05],
#         [2.6565e+05],
#         [1.2951e+05],
#         [1.2037e+05],
#         [1.9337e+02],
#         [2.6116e+04],
#         [1.2902e+02],
#         [1.0059e+01],
#         [1.0059e+01]], grad_fn=<MulBackward0>) tensor([70, 12, 18, 30, 61, 45, 54, 72, 58, 60, 32, 31,  3, 47, 69, 73])
# LABEL:  tensor([[0.0602],
#         [0.0012],
#         [0.0676],
#         [0.1299],
#         [0.0002],
#         [0.0002],
#         [0.0845],
#         [0.0546],
#         [0.0842],
#         [0.0494],
#         [0.0339],
#         [0.0002],
#         [0.0000],
#         [0.0002],
#         [0.0002],
#         [0.0002]])
# ---------------------
# tensor([[221636.0000],
#         [  4488.0000],
#         [248848.0000],
#         [477972.0312],
#         [   812.0000],
#         [   773.0000],
#         [311059.0000],
#         [201021.0000],
#         [309791.0000],
#         [181920.0000],
#         [124703.0078],
#         [   668.0000],
#         [     0.0000],
#         [   680.0000],
#         [   771.0000],
#         [   787.0000]])
# ACC:  tensor(0)
# ----------------------------------
# OUTPUTS:  tensor([[3.4914e-02],
#         [7.4469e-02],
#         [8.5802e-06],
#         [6.7050e-02],
#         [3.2709e-02],
#         [3.9707e-05],
#         [3.9707e-05],
#         [5.2543e-05],
#         [4.5422e-05],
#         [3.3074e-06],
#         [3.9707e-05]], grad_fn=<ViewBackward0>)
# ---------------------
# tensor([[1.2849e+05],
#         [2.7406e+05],
#         [3.1576e+01],
#         [2.4675e+05],
#         [1.2037e+05],
#         [1.4613e+02],
#         [1.4613e+02],
#         [1.9337e+02],
#         [1.6716e+02],
#         [1.2172e+01],
#         [1.4613e+02]], grad_fn=<MulBackward0>) tensor([28, 62, 57,  2, 40, 67, 55, 27, 23, 10, 71])
# LABEL:  tensor([[0.0525],
#         [0.0467],
#         [0.0002],
#         [0.0676],
#         [0.0350],
#         [0.0002],
#         [0.0002],
#         [0.0002],
#         [0.0002],
#         [0.0001],
#         [0.0002]])
# ---------------------
# tensor([[193168.],
#         [171994.],
#         [   805.],
#         [248950.],
#         [128956.],
#         [   652.],
#         [   700.],
#         [   655.],
#         [   609.],
#         [   510.],
#         [   661.]])
# ACC:  tensor(0)
# ----------------------------------
# LOSS TEST:  tensor(0.0223, grad_fn=<AddBackward0>) ================================================================================================================
# CORRECT:  0.9866666666666667
# TIME  0.12766575813293457
# Train tensor(0.0015, grad_fn=<DivBackward0>)
# Test tensor(0.0045, grad_fn=<DivBackward0>)
# saba@saba-VivoBook-ASUSLaptop-X512JP-R564JP:~/DL/time pred$ 

