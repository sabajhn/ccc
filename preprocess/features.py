import numpy as np
import pymetis as metis
import re
import torch
import time
import math

# from numba import jit, cuda

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_time(file_dict):
    # output=[]
    with open(file_dict, "r") as f:
        lines = f.readlines()
        for line in lines:
        # If the line starts with Circuit name, append it to the output list
            if line.startswith("time_route"):
                output=re.findall(r"\d+", line.strip())[0]
    # print("************",output)
    return float(output)
def get_gdata(file_dict):
  with open(file_dict, "r") as f:
    lines = f.readlines()
    n=0
    for num, line in enumerate(lines, 1):
        if 'pin count' in line:
            n=num
    ln=lines[n-1]

    # Create a new list to store the output values
    output = []
    my_list = ["EMPTY" ,"io" ,"PLL" ,"LAB" ,"DSP" ,"M9K" ,"M144K"]
    # Loop through the lines
    for line in lines:
        # If the line starts with Circuit name, append it to the output list
        if line.startswith("Circuit name"):
            output.append(line.strip()) # remove the newline character
        # If the line starts with #######################222##, stop the loop
        elif line.startswith("#######################222##"):
            break
        # Otherwise, split the line by commas and append the values to the output list
        elif(line.startswith("device ut")):
            values = line.split(",")
            output.append("device ut")
            output.append(values[1])
            for x in my_list:
                b = True
                for j in range(len(values)):
                    if(values[j] == x):
                        b = False
                        output.append(values[j])
                        output.append(values[j+1])
                if(b):
                    output.append(x)
                    output.append("0.0")
        else:
            values = line.split(",")
            output.extend(values)

    # Remove any empty strings or newline characters from the output list
    output = [x for x in output if x and x != "\n"]
    # print(output)
    # print([",".join(output)+','+ ln])
    # print("-----------------")
  return [",".join(output)+','+ ln]



def get_circuit_name1(features):
    circuit_names = []
    for i in features:
        i.pop(0)
        circuit_names=i.pop(0)
#         i.pop(0)
    return circuit_names


def feature_values1(features):

    dict_res=[]
    features_val=[]
    fpga_size=[]
    # blk_size=[]
    # edge_size=[]
    for i in features:
        i.pop()
        if 'EMPTY' in i:
            if i[i.index('EMPTY')+1] == '0':
                del i[i.index('EMPTY')+1]
            if i[i.index('EMPTY')+1] == '0':
                del i[i.index('EMPTY')+1]
            if i[i.index('EMPTY')+1] == '0':
                del i[i.index('EMPTY')+1]
            del i [i.index('EMPTY')]
#         print(i)
        # blk_size=int(i[i.index('block size')+1])
        fpga_size.append(int(i[i.index('fpga size')+1]))
        fpga_size.append(int(i[i.index('fpga size')+2]))
        # print(fpga_size)

        # edge_size=int(i[i.index('edge size')+1])
        # print(blk_size)
#         i [i.index('block size')]
#         i [i.index('edge size')]

        numbers = [float(s) for s in i if s.isnumeric() or '.' in s] # get the numbers and convert them to float
#         print(numbers)
        numbers.append(fpga_size)
        # numbers.append(blk_size)
        # numbers.append(edge_size)
        # features_val.append(numbers)
    return numbers

def get_features1(lst):
    features = []
    res=[]
    circuit_name =[]
#     for line in lst:
    features = [s.split(',') for s in lst]

#     data = lst.strip().split(',')
    # print(features)
#     features.append(features)
#     print(features)
    circuit_name = get_circuit_name1(features)
    # print(circuit_name)

#     features = feature_values(features)
    a = feature_values1(features)
    b_sz=a[len(a)-1]
    # e_sz=a[len(a)-2]
    a.pop()
    # a.pop()
    return a,circuit_name,b_sz
    # ,e_sz

def get_routability(file_dict):
    with open(file_dict, "r") as f:
    # Read the lines into a list
        output=[]
        lines = f.readlines()
        ln=lines[0]
        output.append(int(ln))
    return output

def edge_ex(file_dict):

    file_dict = file_dict
    edge = []
    res=[]
    with open(file_dict,'r') as data_file:
        for line in data_file:

            data = line.strip().split(',')
    #         print([data])
            edge.append(np.asarray(data))

    # print(edge)
    edge.pop()
    edge = np.asarray(edge)
    edges = [[int(element) for element in sublist] for sublist in edge]

    return edges

def node_features(file_dict):
    node_f = []
    block_types = {"io":0, "PLL":1, "LAB":2, "DSP":3, "M9K":4, "M144K":5}
    res=[]
    with open(file_dict,'r') as data_file:
        for line in data_file:

            data = line.strip().split(',')
            node_f.append(data[0])
    node_f.pop()
    # print(node_f)

    nodes=[[0,0,0,0,0,0] for i in range(len(node_f))]
    for i in range(len(node_f)):
        nodes[i][block_types[node_f[i]]] += 1

    return nodes
# exit()
def matrix_cong(file_dict,fpga_size):
    # Initialize arr as a 3D array of zeros
    arr = np.zeros((2,fpga_size[0],fpga_size[1]))
    with open(file_dict,'r') as data_file:
        for line in data_file:
            data = line.strip().split(',')
            if int(data[2]) !=0 or int(data[3]) !=0:
                arr[:,int(data[0]),int(data[1])] = [float(data[2]),float(data[3])]
    return arr
# @jit(target_backend='cuda') 

def matrix_finout(file_dict,fpga_size):
    start1=time.time()
    
    arr = np.zeros((fpga_size[0],fpga_size[1]))
    print("1f",time.time() - start1)
    start2=time.time()
    with open(file_dict,'r') as data_file:
        for line in data_file:

            data = line.strip().split(',')
            if "#####4##################" not in line:
                if int(data[2]) != 0: 
                    arr[int(data[0])][int(data[1])] = int(data[2])
    print("2f",time.time() - start2)
    return arr
# @jit(target_backend='cuda') 
def matrix_cong_tagh(file_dict,fpga_size):
    try:


        start1=time.time()
        # arr=[]
        arr = np.zeros((fpga_size[0],fpga_size[1]))
        # arr = torch. empty(fpga_size[0],fpga_size[1])
        print("1c",time.time() - start1)
        start2=time.time()
        
        
        with open(file_dict,'r') as data_file:
            for line in data_file:

                data = line.strip().split(',')
                if "#####4##################" not in line:
                    if int(data[2]) != 0:
                        arr[int(data[0])][int(data[1])] = float(data[2])

        print("2c",time.time() - start2)
    except ValueError:
        start1=time.time()
        # arr=[]
        # arr = np.zeros((fpga_size[0],fpga_size[1]))
        # arr = torch. empty(fpga_size[0],fpga_size[1])
        print("1c",time.time() - start1)
        start2=time.time()
        
        
        with open(file_dict,'r') as data_file:
            for line in data_file:

                data = line.strip().split(',')
                if "#####4##################" not in line:
                    if int(data[2]) != 0:
                        arr[int(data[0])][int(data[1])] = float(data[2])

        print("2c",time.time() - start2)
    except IndexError:
        start1=time.time()
        # arr=[]
        # arr = np.zeros((fpga_size[0],fpga_size[1]))
        # arr = torch. empty(fpga_size[0],fpga_size[1])
        print("1c",time.time() - start1)
        start2=time.time()
        
        
        with open(file_dict,'r') as data_file:
            for line in data_file:

                data = line.strip().split(',')
                if "#####4##################" not in line:
                    if int(data[2]) != 0:
                        arr[int(data[0])][int(data[1])] = float(data[2])

        print("2c",time.time() - start2)
    return arr

# @jit(target_backend='cuda')  
def matrix_coord(file_dict,fpga_size,file_dic1,file_dic2,file_dic3):
    # time.sleep(0.00001)
    try:

        # in_minx,in_miny,in_xmax,in_ymax,out_minx,out_miny,out_xmax,out_ymax=np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1]))
        start1=time.time()
        
        
        in_minx=np.zeros((fpga_size[0],fpga_size[1]))
        in_miny=np.zeros((fpga_size[0],fpga_size[1]))
        in_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        in_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        out_minx= np.zeros((fpga_size[0],fpga_size[1]))
        out_miny=np.zeros((fpga_size[0],fpga_size[1]))
        out_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        out_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        print("1m",time.time() - start1)
        start2=time.time()
        # arr = np.zeros((fpga_size[0],fpga_size[1]))
        with open(file_dict,'r') as data_file:
            for line in data_file:
                data = line.strip().split(',')
                if "#####4##################" not in line:
                        
                    if (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0):

                        in_minx[int(data[0])][int(data[1])] = int(data[2])
                        in_miny[int(data[0])][int(data[1])] = int(data[3])
                        in_xmax[int(data[0])][int(data[1])] = int(data[4])
                        in_ymax[int(data[0])][int(data[1])] = int(data[5])
                        out_minx[int(data[0])][int(data[1])] = int(data[6])
                        out_miny[int(data[0])][int(data[1])] = int(data[7])
                        out_xmax[int(data[0])][int(data[1])] = int(data[8])
                        out_ymax[int(data[0])][int(data[1])] = int(data[9])
        print("2m",time.time() - start2)
    except ValueError:
        # n_ymax,out_minx,out_miny,out_xmax,out_ymax=np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1]))
        start1=time.time()
        
        
        # in_minx=np.zeros((fpga_size[0],fpga_size[1]))
        # in_miny=np.zeros((fpga_size[0],fpga_size[1]))
        # in_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        # in_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        # out_minx= np.zeros((fpga_size[0],fpga_size[1]))
        # out_miny=np.zeros((fpga_size[0],fpga_size[1]))
        # out_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        # out_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        print("1m",time.time() - start1)
        start2=time.time()
        # arr = np.zeros((fpga_size[0],fpga_size[1]))
        with open(file_dict,'r') as data_file:
            for line in data_file:
                data = line.strip().split(',')
                if "#####4##################" not in line:
                        
                    if (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0):
                        # print(int(data[0]))
                        # print(int(data[1]))
                        # print(int(data[2]))
                        # print(int(data[3]))
                        # print(int(data[4]))
                        # print(int(data[5]))
                        # print(int(data[6]))
                        # print(int(data[7]))
                        # print(int(data[8]))
                        # print(int(data[9]))
                        
                        in_minx[int(data[0])][int(data[1])] = int(data[2])
                        in_miny[int(data[0])][int(data[1])] = int(data[3])
                        in_xmax[int(data[0])][int(data[1])] = int(data[4])
                        in_ymax[int(data[0])][int(data[1])] = int(data[5])
                        out_minx[int(data[0])][int(data[1])] = int(data[6])
                        out_miny[int(data[0])][int(data[1])] = int(data[7])
                        out_xmax[int(data[0])][int(data[1])] = int(data[8])
                        out_ymax[int(data[0])][int(data[1])] = int(data[9])
    except IndexError:
            # n_ymax,out_minx,out_miny,out_xmax,out_ymax=np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1]))
            start1=time.time()
            
            
            # in_minx=np.zeros((fpga_size[0],fpga_size[1]))
            # in_miny=np.zeros((fpga_size[0],fpga_size[1]))
            # in_xmax=np.zeros((fpga_size[0],fpga_size[1]))
            # in_ymax=np.zeros((fpga_size[0],fpga_size[1]))
            # out_minx= np.zeros((fpga_size[0],fpga_size[1]))
            # out_miny=np.zeros((fpga_size[0],fpga_size[1]))
            # out_xmax=np.zeros((fpga_size[0],fpga_size[1]))
            # out_ymax=np.zeros((fpga_size[0],fpga_size[1]))
            print("1m",time.time() - start1)
            start2=time.time()
            # arr = np.zeros((fpga_size[0],fpga_size[1]))
            with open(file_dict,'r') as data_file:
                for line in data_file:
                    data = line.strip().split(',')
                    if "#####4##################" not in line:
                            
                        if (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0):
                            # print(int(data[0]))
                            # print(int(data[1]))
                            # print(int(data[2]))
                            # print(int(data[3]))
                            # print(int(data[4]))
                            # print(int(data[5]))
                            # print(int(data[6]))
                            # print(int(data[7]))
                            # print(int(data[8]))
                            # print(int(data[9]))
                            
                            in_minx[int(data[0])][int(data[1])] = int(data[2])
                            in_miny[int(data[0])][int(data[1])] = int(data[3])
                            in_xmax[int(data[0])][int(data[1])] = int(data[4])
                            in_ymax[int(data[0])][int(data[1])] = int(data[5])
                            out_minx[int(data[0])][int(data[1])] = int(data[6])
                            out_miny[int(data[0])][int(data[1])] = int(data[7])
                            out_xmax[int(data[0])][int(data[1])] = int(data[8])
                            out_ymax[int(data[0])][int(data[1])] = int(data[9])
            print("2m",time.time() - start2)
    
    return[in_minx,in_miny,in_xmax,in_ymax,out_minx,out_miny,out_xmax,out_ymax, matrix_finout(file_dic1,fpga_size),matrix_finout(file_dic2,fpga_size),matrix_cong_tagh(file_dic3,fpga_size)]
#  [torch.tensor(in_minx),torch.tensor(in_miny),torch.tensor(in_xmax),torch.tensor(in_ymax),torch.tensor(out_minx),torch.tensor(out_miny),torch.tensor(out_xmax),torch.tensor(out_ymax), torch.tensor(matrix_finout(file_dic1,fpga_size)),torch.tensor(matrix_finout(file_dic2,fpga_size)),torch.tensor(matrix_cong_tagh(file_dic3,fpga_size))]


def matrix_coord1(file_dict,fpga_size,file_dic1,file_dic2,file_dic3,in_minx,in_miny,in_xmax,in_ymax,out_minx,out_miny,out_xmax,out_ymax):
    # time.sleep(0.00001)
    try:

        # in_minx,in_miny,in_xmax,in_ymax,out_minx,out_miny,out_xmax,out_ymax=np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1]))
        start1=time.time()
        
        
        # in_minx=np.zeros((fpga_size[0],fpga_size[1]))
        # in_miny=np.zeros((fpga_size[0],fpga_size[1]))
        # in_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        # in_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        # out_minx= np.zeros((fpga_size[0],fpga_size[1]))
        # out_miny=np.zeros((fpga_size[0],fpga_size[1]))
        # out_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        # out_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        print("1m",time.time() - start1)
        start2=time.time()
        # arr = np.zeros((fpga_size[0],fpga_size[1]))
        with open(file_dict,'r') as data_file:
            for line in data_file:
                data = line.strip().split(',')
                if "#####4##################" not in line:
                        
                    # if ( in_minx[int(data[0])][int(data[1])] != int(data[2])) or (in_miny[int(data[0])][int(data[1])] != int(data[3])) or (in_xmax[int(data[0])][int(data[1])] != int(data[4])) or (in_ymax[int(data[0])][int(data[1])] != int(data[5])) or (out_minx[int(data[0])][int(data[1])] != int(data[6])) or (out_miny[int(data[0])][int(data[1])] != int(data[7])) or (out_xmax[int(data[0])][int(data[1])] != int(data[8])) or (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0) or (out_ymax[int(data[0])][int(data[1])] != int(data[9])):
                    if ( in_minx[int(data[0])][int(data[1])] != int(data[2])) or (in_miny[int(data[0])][int(data[1])] != int(data[3])) or (in_xmax[int(data[0])][int(data[1])] != int(data[4])) or (in_ymax[int(data[0])][int(data[1])] != int(data[5])) or (out_minx[int(data[0])][int(data[1])] != int(data[6])) or (out_miny[int(data[0])][int(data[1])] != int(data[7])) or (out_xmax[int(data[0])][int(data[1])] != int(data[8])) or (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0) or (out_ymax[int(data[0])][int(data[1])] != int(data[9])):

                        in_minx[int(data[0])][int(data[1])] = int(data[2])
                        in_miny[int(data[0])][int(data[1])] = int(data[3])
                        in_xmax[int(data[0])][int(data[1])] = int(data[4])
                        in_ymax[int(data[0])][int(data[1])] = int(data[5])
                        out_minx[int(data[0])][int(data[1])] = int(data[6])
                        out_miny[int(data[0])][int(data[1])] = int(data[7])
                        out_xmax[int(data[0])][int(data[1])] = int(data[8])
                        out_ymax[int(data[0])][int(data[1])] = int(data[9])
        print("2m",time.time() - start2)
    except ValueError:
        # n_ymax,out_minx,out_miny,out_xmax,out_ymax=np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1]))
        start1=time.time()
        
        
        # in_minx=np.zeros((fpga_size[0],fpga_size[1]))
        # in_miny=np.zeros((fpga_size[0],fpga_size[1]))
        # in_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        # in_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        # out_minx= np.zeros((fpga_size[0],fpga_size[1]))
        # out_miny=np.zeros((fpga_size[0],fpga_size[1]))
        # out_xmax=np.zeros((fpga_size[0],fpga_size[1]))
        # out_ymax=np.zeros((fpga_size[0],fpga_size[1]))
        print("1m",time.time() - start1)
        start2=time.time()
        # arr = np.zeros((fpga_size[0],fpga_size[1]))
        with open(file_dict,'r') as data_file:
            for line in data_file:
                data = line.strip().split(',')
                if "#####4##################" not in line:
                        
                    if ( in_minx[int(data[0])][int(data[1])] != int(data[2])) or (in_miny[int(data[0])][int(data[1])] != int(data[3])) or (in_xmax[int(data[0])][int(data[1])] != int(data[4])) or (in_ymax[int(data[0])][int(data[1])] != int(data[5])) or (out_minx[int(data[0])][int(data[1])] != int(data[6])) or (out_miny[int(data[0])][int(data[1])] != int(data[7])) or (out_xmax[int(data[0])][int(data[1])] != int(data[8])) or (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0) or (out_ymax[int(data[0])][int(data[1])] != int(data[9])):

                    # if (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0):
                        # print(int(data[0]))
                        # print(int(data[1]))
                        # print(int(data[2]))
                        # print(int(data[3]))
                        # print(int(data[4]))
                        # print(int(data[5]))
                        # print(int(data[6]))
                        # print(int(data[7]))
                        # print(int(data[8]))
                        # print(int(data[9]))
                        
                        in_minx[int(data[0])][int(data[1])] = int(data[2])
                        in_miny[int(data[0])][int(data[1])] = int(data[3])
                        in_xmax[int(data[0])][int(data[1])] = int(data[4])
                        in_ymax[int(data[0])][int(data[1])] = int(data[5])
                        out_minx[int(data[0])][int(data[1])] = int(data[6])
                        out_miny[int(data[0])][int(data[1])] = int(data[7])
                        out_xmax[int(data[0])][int(data[1])] = int(data[8])
                        out_ymax[int(data[0])][int(data[1])] = int(data[9])
    except IndexError:
            # n_ymax,out_minx,out_miny,out_xmax,out_ymax=np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1])),np.zeros((fpga_size[0],fpga_size[1]))
            start1=time.time()
            
            
            # in_minx=np.zeros((fpga_size[0],fpga_size[1]))
            # in_miny=np.zeros((fpga_size[0],fpga_size[1]))
            # in_xmax=np.zeros((fpga_size[0],fpga_size[1]))
            # in_ymax=np.zeros((fpga_size[0],fpga_size[1]))
            # out_minx= np.zeros((fpga_size[0],fpga_size[1]))
            # out_miny=np.zeros((fpga_size[0],fpga_size[1]))
            # out_xmax=np.zeros((fpga_size[0],fpga_size[1]))
            # out_ymax=np.zeros((fpga_size[0],fpga_size[1]))
            print("1m",time.time() - start1)
            start2=time.time()
            # arr = np.zeros((fpga_size[0],fpga_size[1]))
            with open(file_dict,'r') as data_file:
                for line in data_file:
                    data = line.strip().split(',')
                    if "#####4##################" not in line:
                            
                        if ( in_minx[int(data[0])][int(data[1])] != int(data[2])) or (in_miny[int(data[0])][int(data[1])] != int(data[3])) or (in_xmax[int(data[0])][int(data[1])] != int(data[4])) or (in_ymax[int(data[0])][int(data[1])] != int(data[5])) or (out_minx[int(data[0])][int(data[1])] != int(data[6])) or (out_miny[int(data[0])][int(data[1])] != int(data[7])) or (out_xmax[int(data[0])][int(data[1])] != int(data[8])) or (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0) or (out_ymax[int(data[0])][int(data[1])] != int(data[9])):

                        # if (int(data[2]) != 0) or (int(data[3]) != 0) or (int(data[4]) != 0) or (int(data[5]) != 0) or (int(data[6]) != 0) or (int(data[7]) != 0):
                            # print(int(data[0]))
                            # print(int(data[1]))
                            # print(int(data[2]))
                            # print(int(data[3]))
                            # print(int(data[4]))
                            # print(int(data[5]))
                            # print(int(data[6]))
                            # print(int(data[7]))
                            # print(int(data[8]))
                            # print(int(data[9]))
                            
                            in_minx[int(data[0])][int(data[1])] = int(data[2])
                            in_miny[int(data[0])][int(data[1])] = int(data[3])
                            in_xmax[int(data[0])][int(data[1])] = int(data[4])
                            in_ymax[int(data[0])][int(data[1])] = int(data[5])
                            out_minx[int(data[0])][int(data[1])] = int(data[6])
                            out_miny[int(data[0])][int(data[1])] = int(data[7])
                            out_xmax[int(data[0])][int(data[1])] = int(data[8])
                            out_ymax[int(data[0])][int(data[1])] = int(data[9])
            print("2m",time.time() - start2)
    
    return[in_minx,in_miny,in_xmax,in_ymax,out_minx,out_miny,out_xmax,out_ymax, matrix_finout(file_dic1,fpga_size),matrix_finout(file_dic2,fpga_size),matrix_cong_tagh(file_dic3,fpga_size)]
#  [torch.tensor(in_minx),torch.tensor(in_miny),torch.tensor(in_xmax),torch.tensor(in_ymax),torch.tensor(out_minx),torch.tensor(out_miny),torch.tensor(out_xmax),torch.tensor(out_ymax), torch.tensor(matrix_finout(file_dic1,fpga_size)),torch.tensor(matrix_finout(file_dic2,fpga_size)),torch.tensor(matrix_cong_tagh(file_dic3,fpga_size))]


def finout(file_dict):
    nodeio = []
    res=[]
    with open(file_dict,'r') as data_file:
        for line in data_file:

            data = line.strip().split(',')
            # print(data)
            nodeio.append(np.asarray(data))

    # print(edge)
    nodeio.pop()
    nodeio = np.asarray(nodeio)
    fin=[]
    fout=[]
    xx = 0
    for i in nodeio:
        #     i = i.astype(int)
        a,b = i
        fin.append(int(a))
        fout.append(int(b) - int(xx))
        xx = int(b)
    return fin, fout

def vinout_blks(edges, blks_size, fin, fout):
    fin_blks=[]
    fout_blks=[]
    for k in range(len(edges)): # loop over the subarrays of edges
        fin_blks.append([]) # use nested lists instead of sets
        fout_blks.append([])
        for i in range(blks_size[k]): # loop over the nodes of each subarray
            fin_blks[k].append([])
            fout_blks[k].append([])
        for i, edge in enumerate(edges[k]): # loop over the edges of each subarray
            a,b  = edge
            fin_blks[k][b].append(a) # append instead of add
            fout_blks[k][a].append(b)
    in_blks=[]
    out_blks=[]
    in_out_blks=[]
    out_in_blks=[]
        # Replace sum with variance using np.var function
    for k in range(len(fin)): # loop over the subarrays of fin and fout
      in_blks.append([])
      out_blks.append([])
      in_out_blks.append([])
      out_in_blks.append([])
      for i in range(blks_size[k]): # loop over the nodes of each subarray
        in_blks[k].append(0)
        out_blks[k].append(0)
        in_out_blks[k].append(0)
        out_in_blks[k].append(0)

      for i in range(len(fin_blks[k])): # loop over the fin_blks and fout_blks of each subarray

        my_list = []
        for j in fin_blks[k][i]:
          my_list.append(fin[k][j])
        x = np.sqrt(np.var(my_list))
        if(math.isnan(x)):
            x = 0.0
        in_blks[k][i] = x # use indexing instead of accessing by value and replace sum with variance
        my_list = []
        for j in fout_blks[k][i]:
          my_list.append(fout[k][j])
        x = np.sqrt(np.var(my_list))
        if(math.isnan(x)):
            x = 0.0
        out_blks[k][i] = x # use indexing instead of accessing by value and replace sum with variance

        my_list = []
        for j in fin_blks[k][i]:
          my_list.append(fout[k][j])
        x = np.sqrt(np.var(my_list))
        if(math.isnan(x)):
            x = 0.0
        in_out_blks[k][i] = x # use indexing instead of accessing by value and replace sum with variance
        my_list = []
        for j in fout_blks[k][i]:
          my_list.append(fin[k][j])
        x = np.sqrt(np.var(my_list))
        if(math.isnan(x)):
            x = 0.0
        out_in_blks[k][i] = x # use indexing instead of accessing by value and replace sum with variance

    return in_blks,out_blks,in_out_blks,out_in_blks

def finout_blks(edges, blks_size, fin, fout):
    fin_blks=[]
    fout_blks=[]
    for k in range(len(edges)): # loop over the subarrays of edges
        fin_blks.append([]) # use nested lists instead of sets
        fout_blks.append([])
        for i in range(blks_size[k]): # loop over the nodes of each subarray
            fin_blks[k].append([])
            fout_blks[k].append([])
        for i, edge in enumerate(edges[k]): # loop over the edges of each subarray
            a,b  = edge
            fin_blks[k][b].append(a) # append instead of add
            fout_blks[k][a].append(b)
    in_blks=[]
    in_out_blks=[]
    out_blks=[]
    out_in_blks=[]
    for k in range(len(fin)): # loop over the subarrays of fin and fout
        in_blks.append([])
        in_out_blks.append([])
        out_blks.append([])
        out_in_blks.append([])
        for i in range(blks_size[k]): # loop over the nodes of each subarray
            in_blks[k].append(0)
            in_out_blks[k].append(0)
            out_blks[k].append(0)
            out_in_blks[k].append(0)
        for i in range(len(fin_blks[k])): # loop over the fin_blks and fout_blks of each subarray
            for j in fin_blks[k][i]:
                in_blks[k][i]+=fin[k][j] # use indexing instead of accessing by value
                in_out_blks[k][i]+=fout[k][j]
            for j in fout_blks[k][i]:
                out_blks[k][i]+=fout[k][j]
                out_in_blks[k][i]+=fin[k][j]
    return in_blks,out_blks, in_out_blks, out_in_blks

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def add_metis(edges,pp):
        edge_features=[[] for i in range(len(edges))]

        # G = nx.Graph(edges)
        (edgecuts, parts) = metis.part_graph(pp,adjacency=edges)
        for i in range(len(edges)):
            a,b=edges[i]
            if(parts[a] == parts[b]):
                edge_features[i].append(1)
            else:
                edge_features[i].append(0)
        return  edge_features

def normalize_gf(input):
    # [k,f]
    output = torch.tensor(input)
    output = torch.nn.functional.normalize(output,dim=0,p=2)
    return output

def fine_grained_normalizer(nodef, edgef):
    mx_node =  [0.00001 for i in range(len(nodef[0][0]))]
    mx_edge =  [0.00001 for i in range(len(edgef[0][0]))]
    print(" ||||||||||||| ", mx_node, mx_edge)
    sz = len(nodef)
    for i in range(sz):
        nodef[i] = torch.tensor(nodef[i], dtype=torch.float32)
        edgef[i] = torch.tensor(edgef[i], dtype=torch.float32)
        nf, _ = torch.max(nodef[i],dim=0)
        ef, _ = torch.max(edgef[i],dim=0)
        for j in range(nodef[i].shape[1]):
            mx_node[j] = max(mx_node[j], min(abs(nf[j]),100000.))
        for j in range(edgef[i].shape[1]):
            mx_edge[j] = max(mx_edge[j], min(abs(ef[j]),100000.))

    for i in range(sz):
        for j in range(nodef[i].shape[1]):
            nodef[i][:,j] /= mx_node[j]
            nodef[i] = torch.max(nodef[i],torch.tensor(1.))
        for j in range(edgef[i].shape[1]):
            edgef[i][:,j] /= mx_edge[j]
    print(" || MX ",mx_node)
    print(" || MX ",mx_edge)
    return nodef, edgef

def fine_grained_normalizerr(nodef):
    mx_node =  [0.00001 for i in range(len(nodef[0][0]))]
    # mx_edge =  [0.00001 for i in range(len(edgef[0][0]))]
    # print(" ||||||||||||| ", mx_node, mx_edge)
    sz = len(nodef)
    for i in range(sz):
        nodef[i] = torch.tensor(nodef[i], dtype=torch.float32)
        # edgef[i] = torch.tensor(edgef[i], dtype=torch.float32)
        nf, _ = torch.max(nodef[i],dim=0)
        # ef, _ = torch.max(edgef[i],dim=0)
        for j in range(nodef[i].shape[1]):
            mx_node[j] = max(mx_node[j], min(abs(nf[j]),100000.))
        # for j in range(edgef[i].shape[1]):
            # mx_edge[j] = max(mx_edge[j], min(abs(ef[j]),100000.))

    for i in range(sz):
        for j in range(nodef[i].shape[1]):
            nodef[i][:,j] /= mx_node[j]
            nodef[i] = torch.max(nodef[i],torch.tensor(1.))
        # for j in range(edgef[i].shape[1]):
            # edgef[i][:,j] /= mx_edge[j]
    print(" || MX ",mx_node)
    # print(" || MX ",mx_edge)
    return nodef


def on_off_to_binary(lst):
    # Initialize an empty list to store the output
    output = []
    # Loop through each element in the input list
    for element in lst:
        # If the element is 'on', append 1 to the output list
        if element == ' on':
            output.append(1)
        # If the element is 'off', append 0 to the output list
        elif element == ' off':
            output.append(0)
        # Otherwise, raise an exception
        else:
            raise ValueError('Invalid input')
        # Return the output list
    return output

def vpr_txt(file_dict):
    with open(file_dict,'r') as data_file:
        for line in data_file:

            data = line.strip().split(',')
    #         print(data)
            res=data
    # res.pop()
    # print([[int(x) if x.isdigit() else x for x in sublist] for sublist in res])
    # print(res[4:])
    res[4:]= on_off_to_binary(res[4:])
    res = [int(x) for x in res]
    return res

