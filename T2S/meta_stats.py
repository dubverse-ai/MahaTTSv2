# import math
# from collections import defaultdict
# from config import config


# def calculate_duration(semt):
#     return math.ceil(((len(semt.split()) + 1) / 50) * 2) / 2

# def get_weights(weights,expected_data,languages):
#     new_weights = []

#     expected_weights = config.weights_percentage_duration

#     total_files = sum([expected_data['total'][i] for i in expected_data['total']])

#     duration_multiplier = {i:config.weights_percentage_duration[i]/(expected_data['total'][i]/total_files) for i in config.weights_percentage_duration}
#     print(expected_data['total'],duration_multiplier)
#     for i in weights:
#         new_weights.append(duration_multiplier[i[1]])
#     return new_weights


# def process_file(file_path):
#     weights = []
#     expected_data = defaultdict(lambda: {i:0 for i in ["single_word","5s","10s","15s","20s","20_sentence",">20"]})
#     languages = defaultdict(int)
#     count = 0
#     with open(file_path, 'r') as file:
#         for line in file:
#             count+=1
#             lang, path, text, semt, ref_files = line.split('|')
#             languages[lang]+=1
#             dur = calculate_duration(semt)
#             # weights.append([lang,,1.0])
#             # duration_files['total'][dur] += 1

#             if len(text.strip().split(' '))==1:
#                 expected_data[lang]["single_word"]+=1
#                 expected_data['total']["single_word"]+=1
#                 weights.append([lang,"single_word",1.0])
#                 continue

#             if dur >19.5 and dur<=20:
#                 expected_data[lang]["20_sentence"]+=1
#                 expected_data['total']["20_sentence"]+=1
#                 weights.append([lang,"20_sentence",1.0])
#                 continue

#             if dur<=5:
#                 expected_data[lang]["5s"]+=1
#                 expected_data['total']["5s"]+=1
#                 weights.append([lang,"5s",1.0])
#                 continue
#             elif dur<=10:
#                 expected_data[lang]["10s"]+=1
#                 expected_data['total']["10s"]+=1
#                 weights.append([lang,"10s",1.0])
#                 continue
#             elif dur<=15:
#                 expected_data[lang]["15s"]+=1
#                 expected_data['total']["15s"]+=1
#                 weights.append([lang,"15s",1.0])
#                 continue
#             elif dur<=20:
#                 expected_data[lang]["20s"]+=1
#                 expected_data['total']["20s"]+=1
#                 weights.append([lang,"20s",1.0])
#                 continue
#             else:
#                 # expected_data[lang][">20"]+=1
#                 # expected_data['total'][">20"]+=1
#                 # weights.append([lang,">20",1.0])
#                 continue

#     final_weights = get_weights(weights,expected_data,languages)


#     return final_weights,count

# def process_file_for_heads(file_path,total_processes,process_id):
#     weights = []
#     # heads = defaultdict(lambda: {i:[] for i in ["single_word","5s","10s","15s","20s","20_sentence",">20"]}) # to include langauges
#     heads = {i:[] for i in ["single_word","5s","10s","15s","20s","20_sentence",">20"]}

#     expected_data = defaultdict(lambda: {i:0 for i in ["single_word","5s","10s","15s","20s","20_sentence",">20"]})
#     languages = defaultdict(int)
#     count = 0
#     line_number = -1
#     with open(file_path, 'r') as file:
#         for line in file:
#             count+=1
#             line_number+=1
#             lang, path, text, semt, ref_files = line.split('|')
#             languages[lang]+=1
#             dur = calculate_duration(semt)
#             # weights.append([lang,,1.0])
#             # duration_files['total'][dur] += 1

#             if len(text.strip().split(' '))==1:
#                 expected_data[lang]["single_word"]+=1
#                 expected_data['total']["single_word"]+=1
#                 weights.append([lang,"single_word",1.0])
#                 heads["single_word"].append(line_number)
#                 continue

#             if dur >19.5 and dur<=20:
#                 expected_data[lang]["20_sentence"]+=1
#                 expected_data['total']["20_sentence"]+=1
#                 weights.append([lang,"20_sentence",1.0])
#                 heads["20_sentence"].append(line_number)
#                 continue

#             if dur<=5:
#                 expected_data[lang]["5s"]+=1
#                 expected_data['total']["5s"]+=1
#                 weights.append([lang,"5s",1.0])
#                 heads["5s"].append(line_number)
#                 continue
#             elif dur<=10:
#                 expected_data[lang]["10s"]+=1
#                 expected_data['total']["10s"]+=1
#                 weights.append([lang,"10s",1.0])
#                 heads["10s"].append(line_number)
#                 continue
#             elif dur<=15:
#                 expected_data[lang]["15s"]+=1
#                 expected_data['total']["15s"]+=1
#                 weights.append([lang,"15s",1.0])
#                 heads["15s"].append(line_number)
#                 continue
#             elif dur<=20:
#                 expected_data[lang]["20s"]+=1
#                 expected_data['total']["20s"]+=1
#                 weights.append([lang,"20s",1.0])
#                 heads["20s"].append(line_number)
#                 continue
#             else:
#                 # expected_data[lang][">20"]+=1
#                 # expected_data['total'][">20"]+=1
#                 # weights.append([lang,">20",1.0])
#                 continue

#             line_number+=1
#     # final_weights = get_weights(weights,expected_data,languages)
#     # final_weights = [1]*len(heads) # same weightage
#     if config.ts_gradient_accumulation_steps>1:
#         batch = config.ts_batch_size*total_processes*config.ts_gradient_accumulation_steps//config.ts_num_workers
#     else:
#         batch = config.ts_batch_size*total_processes*config.ts_gradient_accumulation_steps
#     # batch = config.ts_batch_size*total_processes*config.ts_gradient_accumulation_steps
#     # heads = heads[:-1]
#     heads = {i:heads[i] for i in heads if len(heads[i])!=0}
#     total_size = sum([len(heads[i]) for i in heads if len(heads[i])!=0])
#     norm_nums = [len(heads[i])/total_size for i in heads if len(heads[i])!=0]
#     final_weights = []

#     for i in norm_nums:
#         final_weights.append(max(1,math.ceil(i*batch)))

#     rem_elem = sum(final_weights)-batch
#     final_weights[final_weights.index(max(final_weights))]-=rem_elem

#     # heads,final_weights = sorted(zip(heads,final_weights),key=lambda x:x[1])

#     # process_head = []
#     # proc = 0
#     # sm=0
#     # for i in final_weights:
#     #     # sm+=i
#     #     if sm+i >

#     # process_batch_size = config.ts_batch_size*config.ts_gradient_accumulation_steps
#     # proc = 0
#     # lens = {i:len(heads[i]) for i in heads}
#     # while proc <= process_id:
#     #     new_heads ={}
#     #     new_weights =[]
#     #     sm=0
#     #     for i,j in zip(heads,range(len(final_weights))):
#     #         if sm + final_weights[j] > process_batch_size:
#     #             if sm+final_weights[j] == process_batch_size:
#     #                 new_heads[i] = heads[i]
#     #                 new_weights.append(final_weights[j])
#     #                 heads.pop(i)
#     #             else:
#     #                 new_heads[i] = heads[i][:1+(lens[i]*(process_batch_size-sm)//process_batch_size)]
#     #                 heads[i] = heads[i][1+(lens[i]*(process_batch_size-sm)//process_batch_size):]
#     #                 if len(heads[i]) == 0:
#     #                     heads.pop(i)
#     #                 new_weights.append(process_batch_size-sm)
#     #                 final_weights[j]-= process_batch_size-sm
#     #             sm = 0
#     #             proc+=1
#     #             final_weights=final_weights[j:]
#     #             break
#     #         else:
#     #             new_heads[i] = heads[i]
#     #             new_weights.append(final_weights[j])
#     #             heads.pop(i)


#     #     if len(heads) == 0:
#     #         break

#     # print("weights",new_weights,[(i,len(heads[i])) for i in new_heads])
#     # return new_heads,new_weights,count
#     # # make it more effective as to real_batch_size instead of worker_batch_size

#     # # #[867, 31444, 35458, 6764, 1561, 96, 0] per gpu for iitm
#     # # [10,400,400,60,20,1]
#     # #
#     print("weights",final_weights,[(i,len(heads[i])) for i in heads])
#     print(process_id)
#     new_heads, new_weights =  process_batches(heads,final_weights,process_id+1)

#     assert len(new_heads) != 0 and len(new_weights) == len(new_heads), print(new_heads)

#     print("process id",process_id,new_weights,[(i,len(new_heads[i])) for i in new_heads])
#     return new_heads, new_weights, count

# def process_batches(heads, final_weights, process_id=0):
#     if config.ts_gradient_accumulation_steps>1:
#         process_batch_size = config.ts_batch_size * config.ts_gradient_accumulation_steps//config.ts_num_workers
#     else:
#         process_batch_size = config.ts_batch_size * config.ts_gradient_accumulation_steps
#     proc = 0
#     # Create a copy of the original dictionaries to avoid modifying them during iteration
#     remaining_heads = heads.copy()
#     remaining_weights = final_weights.copy()
#     lens = {i: len(heads[i]) for i in heads}

#     while proc <= process_id and remaining_heads:
#         new_heads = {}
#         new_weights = []
#         current_sum = 0

#         # Convert items to list to avoid dictionary size change during iteration
#         items = list(remaining_heads.items())

#         for key, head_list in items:
#             weight = remaining_weights[0]  # Get the corresponding weight

#             if current_sum + weight > process_batch_size:
#                 # Calculate how much of this head we can include
#                 remaining_space = process_batch_size - current_sum
#                 if current_sum + weight == process_batch_size:
#                     # If it fits exactly
#                     new_heads[key] = head_list
#                     new_weights.append(weight)
#                     del remaining_heads[key]
#                     remaining_weights.pop(0)
#                     # print("inside first")
#                 else:
#                     # If we need to split the head
#                     split_point = 1 + (lens[key] * remaining_space) // process_batch_size
#                     new_heads[key] = head_list[:split_point]
#                     remaining_heads[key] = head_list[split_point:]
#                     # print(process_id,"inside >",remaining_heads)
#                     if not remaining_heads[key]:  # If the remaining list is empty
#                         del remaining_heads[key]

#                     new_weights.append(remaining_space)
#                     remaining_weights[0] -= remaining_space
#                     # print(process_id,"inside >",remaining_heads)
#                     # print("inside seciond")
#                 proc += 1
#                 break
#             else:
#                 # If the current head fits completely
#                 new_heads[key] = head_list
#                 new_weights.append(weight)
#                 del remaining_heads[key]
#                 remaining_weights.pop(0)
#                 current_sum += weight
#                 # print("inside third")
#         if len(remaining_heads)==0:
#             proc+=1
#         if proc == process_id:
#             # print("process id",process_id,proc,new_weights,[(i,len(new_heads[i])) for i in new_heads])
#             return new_heads, new_weights

#     return {}, []  # Return empty structures if no valid batch is found
