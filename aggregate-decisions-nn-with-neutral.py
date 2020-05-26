import sys
import json
import os
import os.path
from operator import itemgetter
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

labels = ["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]
classifications = [[], [], []]
ntype = []

parser = argparse.ArgumentParser()
parser.add_argument("--original_jsonl", type=str, required=True)
parser.add_argument("--index_file", type=str, required=True)
parser.add_argument("--nn_file", type=str, required=True)
parser.add_argument("--logprobs_file", type=str, required=True)
parser.add_argument("--submission_file", help="Output", type=str, required=True)
args = parser.parse_args()
original_jsonl = args.original_jsonl
index_file = args.index_file
submission_file = args.submission_file
nn_file = args.nn_file
logprobs_file = args.logprobs_file

jsonl_fp = open(original_jsonl, "r")
index_fp = open(index_file, "r")
logprobs_fp = open(logprobs_file, "r")

if(os.path.exists(submission_file)):
  raise ValueError("Submission file already exists")

out_fp = open(submission_file, "w")

order = []
outputs = {}
support_evidences = {}
refute_evidences = {}
neutral_evidences = {}

jsondecoder = json.JSONDecoder()
jsonencoder = json.JSONEncoder()

for line in tqdm(jsonl_fp.readlines()):
  struct = jsondecoder.decode(line)
  qid = str(struct["id"])
  order.append(qid)   # Output answers in the same order as jsonl input
  outputs[qid] = 1    # Initialize to "not enough info, in case we
                      # didn't actually retrieve any sentences for this qid
  refute_evidences[qid]= []
  support_evidences[qid] = []
  neutral_evidences[qid] = []

print("Collecting FF Decisions (claim level)...")
nn_df = pd.read_csv(nn_file, header=None)
for qid, output in tqdm(nn_df.values):
  output = int(output)
  qid = str(qid)
  if(output != 0 and output != 1 and output != 2):
    raise ValueError("Output format")
  if(not(qid in outputs)):
    raise ValueError("Answer for nonexistent question " + qid)
  outputs[qid] = output

print("Collecting RoBERTa Probabilities (evidence level)...")
index_fp.seek(0)
logprobs_hdr = logprobs_fp.readline()
for output in tqdm(logprobs_fp.readlines()):
    output = output.rstrip().split()
    if len(output)==4:
        output = [float(output[1][1:]), float(output[2]), float(output[3][:-1])]
    elif len(output)==5 and output[1]=='[':
        output = [float(output[2]), float(output[3]), float(output[4][:-1])]
    elif len(output)==5 and output[4]==']':
        output = [float(output[1][1:]), float(output[2]), float(output[3])]
    elif len(output)==6:
        output = [float(output[2]), float(output[3]), float(output[4])]
    
    index = index_fp.readline().rstrip()
    (qid, evidence_number, title, linenum) = index.split()
    decision = np.argmax(output)
    
    if(decision != 0 and decision != 1 and decision != 2):
        raise ValueError("Output format")
    if(not(qid in outputs)):
        raise ValueError("Answer for nonexistent question " + qid)
        
    if(decision == 2):
        refute_evidences[qid].append((output, [title, int(linenum)]))
    elif(decision == 0):
        support_evidences[qid].append((output, [title, int(linenum)]))
    elif(decision == 1):
        neutral_evidences[qid].append((output, [title, int(linenum)]))

print("Writing output...")
for qid in tqdm(order):
  output = outputs[qid]
  label = labels[output]
  struct = {"id": qid, "predicted_label": label}

  evidence = []
  if(output == 0):
    evidence = support_evidences[qid]
  elif(output == 2):
    evidence = refute_evidences[qid]
  
  if len(evidence) >= 5:
      evidence = sorted(evidence, key=lambda x: x[0][output], reverse=True)[:5]
      evidence = [x[1] for x in evidence]
  
  elif (len(evidence) < 5) and (output != 1):
    evidence = [x[1] for x in evidence]
    neutral_evidence = sorted(neutral_evidences[qid], key=lambda x: x[0][output], reverse=True)
    for i in range(5-len(evidence)):
      if len(neutral_evidence) > i:
        evidence.append(neutral_evidence[i][1])
        
  if (len(evidence) < 5) and (output != 1):
    if(output == 0):
      other_evidence = sorted(refute_evidences[qid], key=lambda x: x[0][output], reverse=True)
    elif(output == 2):
      other_evidence = sorted(support_evidences[qid], key=lambda x: x[0][output], reverse=True)
    for i in range(5-len(evidence)):
      if len(other_evidence) > i:
        evidence.append(other_evidence[i][1])

  # sort by line num then title (perform stable sorts in reverse order)
  evidence.sort(key=itemgetter(0))
  evidence.sort(key=itemgetter(1))

  struct["predicted_evidence"] = evidence
  out_fp.write(jsonencoder.encode(struct) + "\n")
