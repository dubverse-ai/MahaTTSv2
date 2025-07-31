# import matplotlib.pyplot as plt
# import numpy as np
# import subprocess
# import json
# from umap import UMAP
# from tqdm import tqdm

# def count_lines_shell(file_path):
#     result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True)
#     return int(result.stdout.split()[0])

# def load_chunk(file_path,chunk_size):
#     lines = count_lines_shell(file_path)
#     with open(file_path,'r') as file:
#         dataset = []
#         # embed = []
#         for i in tqdm(file,total=lines):
#             data = json.loads(i)
#             key = list(data.keys())[0]
#             dataset.append([key,data[key][0]])
#             # embed.append(data[key][1])
#             if len(dataset)==chunk_size:
#                 return dataset
#                 dataset=[]
#                 # embed=[]
#         if len(dataset)!=0:
#             return dataset


# if __name__ == '__main__':

#     file_name = "pocketfm_pure_textlossless_data_stats.json"
#     bs = -1
#     # data = load_chunk(file_name,-1)
#     embed = np.load("/nlsasfs/home/dubverse/varshulg/work/NeuralSpeak/T2S/pocketfm_embeddings.npy")
#     print(embed.shape)
#     plt.scatter(embed[:,0],embed[:,1])
#     # plt.imsave("gst_embed.png")
#     plt.savefig('gst_embed_pocketfm.png')#, dpi=300, bbox_inches='tight')


