#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:12:19 2025

@author: gongxue
"""
import glob

from rdkit import Chem

from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

from rdkit.Chem import MACCSkeys
import numpy as np
from sklearn.manifold import TSNE
from rdkit.Chem import rdFingerprintGenerator


#smiles = "CCO"  # 乙醇
shapes = ['cubic', 'orthorhombic', 'tetragonal']
fingerprint = "MACC"

for shape in shapes:

    flist = glob.glob('./MAPbX3_pdb/'+shape+'/*09[0-9][0-9].pdb')
     
    molecules = [Chem.MolFromPDBFile(file, removeHs=False) for file in flist]
    
    # Generate ECFP4 fingerprints
    radius = 2  # Equivalent to ECFP4
    fp_size = 1024  # Length of the fingerprint vector
    
    if fingerprint == "ECFP":
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size) for mol in molecules]
    
    elif fingerprint == "MACC":
        fingerprints = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules]
    
    
    # Convert RDKit ExplicitBitVect to numpy array
    fingerprint_array = np.array([np.asarray(fp) for fp in fingerprints])
    
    # Initialize t-SNE with desired parameters
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    
    # Apply t-SNE to the fingerprint array
    tsne_results = tsne.fit_transform(fingerprint_array)
    
    values = TSNE(n_components=2, verbose=2).fit_transform(fingerprint_array)
    
    
    frd = 0
    frs = 100
    
    plt.scatter(values[:(frs-frd),0], values[:(frs-frd),1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=40, label="Br")
    plt.scatter(values[(frs-frd):2*(frs-frd),0], values[(frs-frd):2*(frs-frd),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=40, label="Cl")
    plt.scatter(values[2*(frs-frd):3*(frs-frd),0], values[2*(frs-frd):3*(frs-frd),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=40, label="I")
    
    plt.legend(ncol=1, loc='upper right', handlelength=.5, borderpad=.25, fontsize=10, bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.legend(fontsize=12)  # Adjust the fontsize here
    plt.savefig(f"tsne_{fingerprint}_3_clusters_{shape}.png", dpi=200)
    plt.show()
    
