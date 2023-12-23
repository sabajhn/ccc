import os
import torch
import numpy as np
import time
import onnx
import onnxruntime
from preprocess.features import to_numpy, matrix_coord, get_gdata, get_features1
from model.cnn_time import CNN_croute
file_address_vtr="/home/saba/DL/time pred/dataset/blif-res/stratixiv_arch.timing.xml/alu4.blif/1/0"
# "/media/saba/Untitled/res-aware/res-time/ex1010"

#     res1 =[line.strip().split(',') for line in data_file][0]

import torch
import torch.nn as nn
import torch.nn.functional as F



# file_address_vtr =  "localhome/msa417/Desktop/project/vtr-verilog-to-routing/model-cong-aware"
#file_address_vtr =  "/home/saba/verilog_projects/res-msh/vtr-verilog-to-routing/model-cong-aware-bigk-ts"
# model-cong-aware4"
#
file_address_model="./"
# file_address_vtr= "/home/saba/DL/time pred/dataset/blif-res/stratixiv_arch.timing.xml/alu4.blif/1/0"

        
# while True:
    # time.sleep(0.00001)
    #file_address_vtr=""
    #with open(file_address_vtr1 + "blif.txt" ,'r') as data_file:
       
     #     file_address_vtr = data_file.read().strip()
          #print(file_address_vtr)
    #print(file_address_vtr)
    #print(file_address_vtr + "/features/check.txt")
    # if(os.path.isfile(file_address_vtr + "/features/check.txt") == True and os.path.isfile(file_address_vtr + "/out.txt")== False):

start = time.time()
gf1=[]
a,b,f= get_features1(get_gdata(file_address_vtr + "/features/graph_features.txt"))

a.pop(0)
a.pop(0)
gf1.append(a)
print("1",time.time() - start)
start1=time.time()

input_cnn=matrix_coord(file_address_vtr +"/features/coord.txt",f, file_address_vtr + "/features/nodef-fin.txt",file_address_vtr + "/features/nodef-fout.txt",file_address_vtr + "/features/cong_tagh.txt")

print("2",time.time() - start1)
start2=time.time()

with open(file_address_model + "gf5.txt",'r') as data_file:
    res =[line.strip().split(',') for line in data_file][0]
print("3",time.time() - start2)
start3=time.time()
# 
res = np.asarray([float(x) for x in res] ,dtype=float)
print("4",time.time() - start3)
start4=time.time()
# gf1=gf1[:,:]/res
gf1=np.divide(gf1,res)
gf1 = torch.tensor(gf1,dtype=float)
print("5",time.time() - start4)
start5=time.time()
# print("=====================================================")
print("6",time.time() - start4)
start6=time.time()
with open(file_address_model + "inp5.txt" ,'r') as data_file:
    res1 =[line.strip().split(',') for line in data_file][0]

print("7",time.time() - start6)
start7=time.time()
res1 = np.asarray([float(x) for x in res1] ,dtype=float)
# input_cnn = input_cnn[:,:,:]/res1[:]
input_cnn=[input_cnn[i]/float(res1[i]) for i in range(len(res1))]
print("8",time.time() - start7)
start8=time.time()

input_cnn = np.reshape(input_cnn,(1,11, 81, 60)) 
# input_cnn = np.divide(input_cnn, res1)
# print()
input_cnn = torch.tensor(input_cnn)
# input_cnn = torch.stack((input_cnn), dim=0).view(1,11, 81, 60)
feat_t = time.time()
print("9",time.time() - start8)
start9=time.time()

start_m = time.time()

model=CNN_croute() 
model.load_state_dict(torch.load(file_address_model + "/blif_time7.pt"))
model.eval()
output = model(input_cnn,gf1)
print(time.time() - start_m,"mmmmmmm")
start11=time.time()

# onnx_program = torch.onnx.dynamo_export(model, input_cnn,gf1)
# # onnx_program.save("./my_image_classifier.onnx")
# print(time.time() - start11)
# start12=time.time()
# onnx_model = onnx.load("./my_image_classifier.onnx")
# # onnx.checker.check_model(onnx_model)
# print(time.time() - start12,"A")
# start13=time.time()
# onnx_input = onnx_program.adapt_torch_inputs_to_onnx(input_cnn,gf1)
# # print(f"Input length: {len(onnx_input)}")
# # print(f"Sample input: {onnx_input}")
# print(time.time() - start13,"B")
# start14=time.time()
# print(onnxruntime.get_available_providers()) 
# ort_session = onnxruntime.InferenceSession("./my_image_classifier.onnx", providers=['CUDAExecutionProvider'])
# print(time.time() - start14,"C")
# start15=time.time()
# onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
# print(time.time() - start15,"D")
# start16=time.time()
# onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
# print(time.time() - start16,"E")
# start17=time.time()
# print(onnxruntime_outputs)
# print(time.time() - start11,"F")
exit()

torch_outputs = model(input_cnn,gf1)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")

print(f"Sample output: {onnxruntime_outputs}")
print(time.time() - start11)
# start12=time.time()
print(torch_outputs)
# print(time.time() - start12)
exit()
output = model(input_cnn,gf1)
model_end = time.time()
print(output)

start_f = time.time()
f = open(file_address_vtr + "/out.txt", "w")
f.write(str(float(output)))
f.close()
end_f = time.time()




print("FEATURE TIME ", feat_t - start)
print("MODEL TIME ", model_end - start_m)
print("FILE TIME ",end_f - start_f)
print("ALLL TIME ",end_f - start)
    # if(os.path.isfile(file_address_vtr + "/out1.txt") == False and os.path.isfile(file_address_vtr + "/features/check.txt") == True and os.path.isfile(file_address_vtr+"/out.txt")== True):
    #     f1 = open(file_address_vtr + "/out1.txt", "w")
    #     f1.write("1")
    #     f1.close()
        #os.remove(file_address_vtr)

