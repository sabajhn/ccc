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

# file_names  = os.listdir("/home/saba/verilog_projects/inp/vtr-verilog-to-routing/vtr_flow/benchmarks/blif/ss/alu4/")
file_address="/home/saba/verilog_projects/inp/vtr-verilog-to-routing/vtr_flow/benchmarks/blif/ss/alu4/"
# folder_names = []
# # print(file_names)
# for name in file_names:
#     # abs_path = os.path.abspath(name)
#     # # folder_names.append(abs_path)
#     # print(abs_path)
#     if("alu4_" in name):
#         # folder_names.append(abs_path)
#     #     name1 = abs_path.replace(".blif","")
#     #     print(name1)
#         folder_names.append("/home/saba/verilog_projects/inp/vtr-verilog-to-routing/vtr_flow/benchmarks/blif/ss/alu4/"+name)
#     #     print(folder_names)

# file_address="/content/drive/MyDrive/dataset/blif-res2/stratixiv_arch.timing.xml"
# file_address= "/media/saba/Untitled/cong_mh/vtr-verilog-to-routing-congestion_placement/P

# file_address ="/home/saba/verilog_projects/congestion/vtr-verilog-to-routing/res-aware/res-cong/stratixiv_arch.timing.xml"
# "/media/saba/Untitled/congest"
# "/home/saba/verilog_projects/res-msh/vtr-verilog-to-routing/cong-res3/"
# file_address="/home/saba/DL/time pred/dataset/blif-res2/stratixiv_arch.timing.xml/"
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
blif=["alu4_num_1","alu4_num_2","alu4_num_3"]
# blif=["alu4.blif","des.blif","s38584.1.blif","bigkey.blif","clma.blif","dsip.blif","misex3.blif","seq.blif","elliptic.blif","clock_aliases.blif","apex4.blif"]
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
f = np.empty([5,len(blif),2],dtype=int)
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

for name in range(len(blif)):
            print(file_address + "/"+ blif[name] + "/" )
            circuit = get_gdata(file_address + "/" + blif[name]  + "/features/graph_features.txt")
            a,b,c= get_features1(circuit)
            print(c)
            if c==[81,60]:
            # if c==[23,17]:
              print(file_address + "/"+ blif[name] + "/" )

              # print(file_address + "/"+ blif[name] + "/" )
              routability.append(get_routability(file_address + "/" + blif[name]   +"/features/routability.txt"))   
                  
              time_r.append(get_time(file_address + "/"+ blif[name]  + "/features/time.txt"))
              # f[ch][name]=[81,60]
              # f[ch][name]=[194,144]
              print(file_address + "/"+ blif[name] + "/" )
              # for i in range(1,n+1):
              c1,c2,c3,c4,c5,c6,c7,c8= matrix_coord(file_address + "/"+ blif[name] + "/"  +"/features/coord.txt",c)
              # ,fpga_size[i])
              print("------\n")
              # print(c1)
              in_minx[name]=torch.tensor(c1)
              in_minx1= max(in_minx1,in_minx[name].max())
              data = in_minx[name].numpy()
              in_miny[name]=torch.tensor(c2)
              in_miny1 = max(in_miny1,in_minx[name].max())
              data = in_miny[name].numpy()
              in_xmax[name]=torch.tensor(c3)
              in_xmax1= max(in_xmax1,in_xmax[name].max())
              data = in_xmax[name].numpy()
              in_ymax[name]=torch.tensor(c4)
              in_ymax1 = max(in_ymax1,in_ymax[name].max())
              data = in_ymax[name].numpy()
              out_minx[name]=torch.tensor(c5)
              out_minx1 = max(out_minx1,out_minx[name].max())
              data = out_minx[name].numpy()
              out_miny[name]=torch.tensor(c6)
              out_miny1 = max(out_miny1,out_miny[name].max())
              data = out_miny[name].numpy()
              out_xmax[name]=torch.tensor(c7)
              out_xmax1 = max(out_xmax1,out_xmax[name].max())
              data = out_xmax[name].numpy()
              out_ymax[name]=torch.tensor(c8)
              out_ymax1 = max(out_ymax1,out_ymax[name].max())
              data = out_ymax[name].numpy()
              fin[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/"  +"/features/nodef-fin.txt",c))
              fin1 =  max(fin1,fin[name].max())
              
              data = fin[name].numpy()
              fout[name]=torch.tensor(matrix_finout(file_address + "/"+ blif[name] + "/"  + "/features/nodef-fout.txt",c))
              fout1 = max(fout1,fout[name].max())
              # fout1 = max(fout1,fout[name].max())
              data = fout[name].numpy()
              cong_tagh[name]=torch.tensor(matrix_cong_tagh(file_address + "/"+ blif[name] + "/" + "/features/cong_tagh.txt",c))
              # ss.append(get_cong(cong_tagh[name]))
              # print(cong_tagh)
              cong_tagh1 = max(cong_tagh1,cong_tagh[name].max())
              data = cong_tagh[name].numpy()
              # plt.imshow(data, cmap='viridis')
              # plt.savefig(file_address + "/cong_tagh.png")
              input_cnn.append(torch.stack((in_minx[name],in_miny[name],in_xmax[name],in_ymax[name],out_minx[name],out_miny[name],out_xmax[name],out_ymax[name],fin[name],fout[name],cong_tagh[name]),dim=0))
              print(file_address + "/" + blif[name] )
              if(len(a) == 31):
                # print(a)
                # print(get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
                a.insert(0,get_routability(file_address + "/" + blif[name] + "/"+ str(ch)  + "/" +str(i) +"/features/cw.txt")[0])
                # print("--------\n",a)
                a.pop(0)
                a.pop(0)
                a.append(get_cong(cong_tagh[name]))
                a.append(wl(file_address + "/" + blif[name]  +"/vpr_stdout.log"))
                
                # a.append()
                gf1.append(a)
              
              else:
                a.pop(0)
                a.pop(0)
                a.append(get_cong(cong_tagh[name]))
                a.append(wl(file_address + "/" + blif[name] +"/vpr_stdout.log"))
                
                gf1.append(a)
            # gf1.pop(0)

res=[]
print(gf1,"gfffffff")
file_address="/home/saba/DL/time pred/"
# ######################################inp train norm##################################333333333333
max_inp = [in_minx1,in_miny1,in_xmax1,in_ymax1,out_minx1,out_miny1,out_xmax1,out_ymax1,fin1,fout1,cong_tagh1]

# max_inp = [in_minx1,in_miny1,in_xmax1,in_ymax1,fin1,fout1,cong_tagh1]

print(len(input_cnn))

# print(len(input_cnn[0]))
# print(len(input_cnn[0][0]))
# print( cong_tagh1)

for i in range(len(max_inp)):
   if max_inp[i] == 0 :
  #  and  max_inp[i] <= 1:
      max_inp[i] =1

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

time_r = torch.tensor(time_r)
print("time_r")
for i in time_r:
   print(i)
print("------------+++++++++++----------------------")
time_max = time_r/time_r.max()
print(time_r.max())
# exit()
inputt=[]

time_rr = time_r

# print(torch.isnan(inputt).sum())
# exit()
out=[]
# # # # print(gf1.shape, inputt.shape)
model=CNN_croute() 

print("shapepe", gf1.shape)

import random

dataas=[]
for i in range(len(fpga_size)):
   dataas.append(i)

random.shuffle(dataas)
train_indices = dataas[int(len(fpga_size)*0.8):]
test_indices = dataas[:int(len(fpga_size)*0.8)]


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