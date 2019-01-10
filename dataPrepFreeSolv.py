
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
from rdkit.Chem import Draw
#get_ipython().run_line_magic('matplotlib', 'inline')

homedir = os.path.expanduser("~/")
homedir = homedir+"chemnetEuler/data/"
df = pd.read_csv(homedir+"freesolv.csv", sep=',')


# In[22]:


#df = df.drop('Unnamed: 0', 1)
#df


# In[23]:


# Add unique alphanumeric identifier
df['id'] = range(1, len(df.index)+1)
df['id'] = 'molid' + df['id'].astype(str)
print(df.shape)
df.columns


# In[24]:


# Remove extraneous SMILES entry only for tox21
#df = df.join(df['smiles'].str.split(' ', 1, expand=True).rename(columns={0:'pre_smiles', 1:'Extraneous_SMILES'}))
#df


# # Check For Invalid Smiles

# In[25]:


# Check for invalid SMILES
mol_list = [Chem.MolFromSmiles(x) for x in df['smiles']]
invalid = len([x for x in mol_list if x is None])
print("No. of invalid entries: "+str(invalid))


# In[26]:


#mol_list = []
#for x in df['pre_smiles']:
#    if Chem.MolFromSmiles(x) == None:
#        print(x)
#        df = df[df.pre_smiles != x]
#    else:
#        mol_list.append(Chem.MolFromSmiles(x))


# # Deal With Duplicate Entries

# In[27]:


mask = df.duplicated('smiles', keep=False)


# In[28]:


#Separate out unique and duplicate entries
df_uni = df[~mask]
df_dup = df[mask]
print(df.shape, df_uni.shape, df_dup.shape)


# In[29]:


# Compute mean of duplicate entries
avg_df = df_dup.groupby('smiles', as_index=False).mean()
avg_df.head(25)


# In[30]:


# Match up average predictions to SMILES and drop duplicate entries
print(df_dup.shape)
df_dup = df_dup.drop(['expt', 'calc'], axis=1)
df_dup = pd.merge(df_dup, avg_df, how="right", on=["smiles"])
print(df_dup.shape)
df_dup = df_dup.drop_duplicates(subset=['smiles'], keep="first")
print(df_dup.shape)


# In[31]:


# Add reliable averaged de-duplicated entries back to unique entries
df2 = pd.concat([df_dup, df_uni], axis=0)
print(df2.shape)
print(df2.smiles.unique().shape)
print(df.smiles.unique().shape)


# In[32]:


# Reset index of df
df2 = df2.reset_index(drop=True)
df2.columns
#df2 = df2.drop('Unnamed: 0',1)
#df2 = df2.drop('Unnamed: 0_x',1)
#df2 = df2.drop('Unnamed: 0_y',1)
df2.head(5)


# In[33]:


print(df2.shape)
df2.to_csv(homedir+"freesolv_all.csv", index=False)


# # Internal Set

# In[34]:


df = pd.read_csv(homedir+"freesolv_all.csv")


# In[35]:


#construct internal test set
size = 0.10
seed = 6
np.random.seed(seed)


# In[36]:


msk = np.random.rand(len(df)) < 0.1
df_tv = df[~msk]
df_int = df[msk]
print(df.shape, df_tv.shape, df_int.shape)


# In[37]:


df_tv.to_csv(homedir+'freesolv_all_trainval.csv', index=False)
df_int.to_csv(homedir+'freesolv_all_int.csv', index=False)


# # split data by task

# In[39]:


# currently one measurement, flash point
# Check for missing labels
dfInt = pd.read_csv(homedir+"freesolv_all_int.csv")
dfInt['calc'].isnull().sum()
df.shape


# In[41]:


#drop data if this is anything greater than 1
df1Int = dfInt[['id','iupac','smiles','expt']]
df1Int.to_csv(homedir+"freesolv_int_expt.csv", index=False)
df2Int = dfInt[['id','iupac','smiles','calc']]
df2Int.to_csv(homedir+"freesolv_int_calc.csv", index=False)
#df1Int.groupby('flashPoint').count()
#df1Int.shape


# In[45]:


dfTrainval = pd.read_csv(homedir+"freesolv_all_trainval.csv")
#print(dfTrainval.head(5))
dfTrainval = dfTrainval[['id','iupac','smiles','expt']]
dfTrainval.to_csv(homedir+"freesolv_tv_expt.csv", index=False)
#dfTrainval.groupby('flashPoint').count()
#dfTrainval.shape
dfTrainval2 = pd.read_csv(homedir+"freesolv_all_trainval.csv")
dfTrainval2 = dfTrainval2[['id','iupac','smiles','calc']]
dfTrainval2.to_csv(homedir+"freesolv_tv_calc.csv", index=False)


# # Prep 2D images

# In[46]:


homedir = os.path.expanduser("~/")
archdir = homedir+"chemnetEuler/archive/"
homedir = homedir+"chemnetEuler/data/"


# In[47]:


from chem_scripts import cs_compute_features, cs_set_resolution, cs_coords_to_grid, cs_check_grid_boundary
from chem_scripts import cs_channel_mapping, cs_map_atom_to_grid, cs_map_bond_to_grid, cs_grid_to_image


# In[48]:


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


# In[49]:


dim = 40       # Size of the box in Angstroms, not radius!
res = 0.5      # Resolution of each pixel
rep = "std"    # Image representation used
nskip = 500    # How many steps till next visualization

gridsize = int(dim/res)


# In[50]:


# Specify dataset name
jobname = "freesolv_int"
taskname = ["calc", "expt"]

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


# In[51]:


# Specify dataset name
jobname = "freesolv_tv"
taskname = ["expt", "calc"]

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


# In[83]:


#m = Chem.MolFromSmiles('F[Si-2](F)(F)(F)(F)F.[NH4+].[NH4+]')
#Draw.MolToImage(m)

