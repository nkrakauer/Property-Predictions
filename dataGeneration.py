#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
from rdkit.Chem import Draw
#get_ipython().run_line_magic('matplotlib', 'inline')

homedir = os.path.expanduser("~/")
homedir = homedir+"property-predictions/dataScraped/"
df = pd.read_csv(homedir+"chenPlusGelest.csv", sep=',')


# In[20]:


df.shape


# In[21]:


# Add unique alphanumeric identifier
df['id'] = range(1, len(df.index)+1)
df['id'] = 'molid' + df['id'].astype(str)
print(df.shape)
df.columns


# In[22]:


df.to_csv(homedir+"data2_all.csv", index=False)


# # Internal Set

# In[23]:


df = pd.read_csv(homedir+"data2_all.csv")


# In[24]:


#construct internal test set
size = 0.10
seed = 6
np.random.seed(seed)


# In[25]:


msk = np.random.rand(len(df)) < 0.1
df_tv = df[~msk]
df_int = df[msk]
print(df.shape, df_tv.shape, df_int.shape)


# In[26]:


df_tv.to_csv(homedir+'data2_all_trainval.csv', index=False)
df_int.to_csv(homedir+'data2_all_int.csv', index=False)


# # Split Data By Task

# In[27]:


# Check for missing labels
dfInt = pd.read_csv(homedir+"data2_all_int.csv")
dfInt['flashPoint'].isnull().sum()
#df.shape


# In[28]:


df1Int = dfInt[['id','compound','smiles','flashPoint']]
df1Int.to_csv(homedir+"data2_int_flashPoint.csv", index=False)


# In[29]:


dfTrainval = pd.read_csv(homedir+"data2_all_trainval.csv")
#print(dfTrainval.head(5))
dfTrainval = dfTrainval[['id','compound','smiles','flashPoint']]
dfTrainval.to_csv(homedir+"data2_tv_flashPoint.csv", index=False)


# ## 2D Images

# In[30]:


homedir = os.path.expanduser("~/")
archdir = homedir+"property-predictions/archive/"
homedir = homedir+"property-predictions/dataScraped/"


# In[31]:


from chem_scripts.skunkImage import cs_compute_features, cs_set_resolution, cs_coords_to_grid, cs_check_grid_boundary
from chem_scripts.skunkImage import cs_channel_mapping, cs_map_atom_to_grid, cs_map_bond_to_grid, cs_grid_to_image


# In[32]:


def gen_image():
    
    exclusion_list = []
    full_array_list = []

    for i in range(0,df.shape[0]):

        # Extract SMILES string
        smiles_string = df["smiles"][i]
        #print(i, smiles_string)

        # Extract ID of molecule
        id_string = df["id"][i]

        # Read SMILES string
        mol = Chem.MolFromSmiles(smiles_string)
        
        # Compute properties
        print(smiles_string)
        mol, df_atom, df_bond, nancheckflag = cs_compute_features(mol)
        
        # Intialize grid
        myarray = cs_set_resolution(gridsize, representation=rep)

        # Map coordinates to grid
        df_atom, atomcheckflag = cs_coords_to_grid(df_atom, dim, res)
        
        # Check if outside grid
        sizecheckflag = cs_check_grid_boundary(df_atom, gridsize)

        if sizecheckflag == True or atomcheckflag == True or nancheckflag == True:

            exclusion_list.append(id_string)
            print("EXCLUSION for "+str(id_string))
            #print('exlusion')

        else:
            # Initialize channels
            channel = cs_channel_mapping()

            # Map atom to grid
            myarray = cs_map_atom_to_grid(myarray, channel, df_atom, representation=rep)

            # Map bond to grid
            myarray = cs_map_bond_to_grid(myarray, channel, df_atom, df_bond, representation=rep)

            # Visualize status every 1000 steps
            #if (i+1)%nskip==0:
               # print("*** PROCESSING "+str(i+1)+": "+str(id_string)+" "+str(smiles_string))
               # cs_grid_to_image(myarray, mol)

            # Generate combined array of raw input
            curr_array = myarray.flatten()
            curr_array_list = curr_array.tolist()
            full_array_list.append(curr_array_list)

    full_array = np.asarray(full_array_list)
    print(full_array.shape)
    print(exclusion_list)

    return(full_array, exclusion_list)


# In[35]:


dim = 40       # Size of the box in Angstroms, not radius!
res = 0.5      # Resolution of each pixel
rep = "engD"    # Image representation used
nskip = 500    # How many steps till next visualization

gridsize = int(dim/res)


# In[36]:


# Specify dataset name
jobname = "data2_int"
taskname = ["flashPoint"]

for task in taskname:

    print("PROCESSING TASK: "+str(jobname)+" "+str(task))
    
    # Specify input and output csv
    filein  = homedir+jobname+"_"+task+".csv"
    print(filein)
    fileout = homedir+jobname+"_"+task+"_image.csv"
    print(fileout)
    # Specify out npy files
    fileimage = archdir+jobname+"_"+task+"_img_"+rep+".npy" 
    print(fileimage)
    filelabel = archdir+jobname+"_"+task+"_img_label.npy" 
    print(filelabel)
    # Generate image
    df = pd.read_csv(filein)
    print(df.columns)
    full_array, exclusion_list = gen_image()
    
    # Dataset statistics before and after image generation
    print("*** Database Specs:")
    print(df.shape[0], len(exclusion_list), int(df.shape[0])-int(len(exclusion_list)))
    
    # Create csv of final data (after exclusion)
    print("*** Separating Database:")
    mod_df = df[~df["id"].isin(exclusion_list)]
    mod_df.to_csv(fileout, index=False)

    # Save generated images as npy
    np.save(fileimage, full_array)
    print(full_array.shape)
    
    # Save labels as npy
    label_array = mod_df[task].as_matrix().astype("float32")
    np.save(filelabel, label_array)
    print(label_array.shape)


# In[ ]:


# Specify dataset name
jobname = "data2_tv"
taskname = ["flashPoint"]

for task in taskname:

    print("PROCESSING TASK: "+str(jobname)+" "+str(task))
    
    # Specify input and output csv
    filein  = homedir+jobname+"_"+task+".csv"
    fileout = homedir+jobname+"_"+task+"_image.csv"
    
    # Specify out npy files
    fileimage = archdir+jobname+"_"+task+"_img_"+rep+".npy" 
    filelabel = archdir+jobname+"_"+task+"_img_label.npy" 
    
    # Generate image
    df = pd.read_csv(filein)
    full_array, exclusion_list = gen_image()
    
    # Dataset statistics before and after image generation
    print("*** Database Specs:")
    print(df.shape[0], len(exclusion_list), int(df.shape[0])-int(len(exclusion_list)))
    
    # Create csv of final data (after exclusion)
    print("*** Separating Database:")
    mod_df = df[~df["id"].isin(exclusion_list)]
    mod_df.to_csv(fileout, index=False)

    # Save generated images as npy
    np.save(fileimage, full_array)
    print(full_array.shape)
    
    # Save labels as npy
    label_array = mod_df[task].as_matrix().astype("float32")
    np.save(filelabel, label_array)
    print(label_array.shape)
