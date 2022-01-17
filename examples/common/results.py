import hashlib
import os
import json
import pickle as pk
import pandas as pd
import numpy as np

def hash_namespace(ns):
    nsdict = vars(ns)
    nsdict.pop('func', None) #can't hash function objects
    return hashlib.md5(json.dumps(nsdict, sort_keys=True).encode('utf-8')).hexdigest()

def check_exists(arguments, results_folder = 'results/', log_file = 'manifest.csv'):
    matching_hash = find_matching(vars(arguments), results_folder, log_file)
    if os.path.exists(os.path.join(results_folder, matching_hash+'.csv')):
        return True
    return False

def find_matching(to_match, results_folder = 'results/', log_file = 'manifest.csv'):
    # load the manifest
    with open(os.path.join(results_folder, log_file), 'r') as f:
        manifest = f.readlines()
    # split each manifest line into [hash, args_string]
    manifest = [ line.split(':', 1) for line in manifest]
    # find matching manifest lines
    matching_hash = None
    for line in manifest:
        str_args = line[1].strip()
        args_dict = json.loads(str_args)
        to_match_tmp = {key : val for (key, val) in to_match.items() if key in args_dict}
        if to_match_tmp == args_dict:
            if matching_hash is not None:
                raise ValueError(f"ERROR: found two matches for arguments dict. Hash 1: {matching_hash} Hash 2: {line[0].strip()} to_match = {to_match}")
            matching_hash = line[0].strip()

    return matching_hash

def load_matching(to_match, results_folder = 'results/', log_file = 'manifest.csv'):
    print("Plot: Matching arguments setting {to_match}")
    matching_hash = find_matching(to_match, results_folder, log_file)
    if matching_hash is None:
        raise ValueError(f"ERROR: no matches for plotting. to_match = {to_match}")
    df = pd.read_csv(os.path.join(results_folder, matching_hash+".csv"))
    print("Plotting data in dataframe:")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    return df

def save(arguments, results_folder = 'results/', log_file = 'manifest.csv', **kwargs):

    # convert the arguments namespace to a dictionary
    nsdict = vars(arguments)

    # remove any names in the nsdict that appear in the "output" (i.e. "data") variables
    for kw, val in kwargs.items():
        nsdict.pop(kw, None)

    # remove the 'func' argument if its there (cant hash function objects)
    nsdict.pop('func', None)

    # hash the input arguments
    arg_hash = hashlib.md5(json.dumps(nsdict, sort_keys=True).encode('utf-8')).hexdigest()

    #make the results folder if it doesn't exist
    if not os.path.exists(results_folder):
      os.mkdir(results_folder)

    # if the file doesn't already exist, create the df file and append a line to manifest
    if not os.path.exists(os.path.join(results_folder, arg_hash+'.csv')):
        with open(os.path.join(results_folder, log_file), 'a') as f:
            manifest_line = arg_hash+':'+ json.dumps(nsdict, sort_keys=True) + '\n'
            f.write(manifest_line)

    #add the output variables back into the dict
    for kw, val in kwargs.items():
        nsdict[kw] = [val]

    #save the df, overwriting a previous result
    df = pd.DataFrame(nsdict)
    df.to_csv(os.path.join(results_folder, arg_hash+'.csv'), index=False)

