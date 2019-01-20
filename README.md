## Property-Predictions :zap::battery::zap:
Neural Network that predicts flashpoint for organosilicon compounds. We convert images of molecules into 4 channel images encoded with basic chemical properties. 

## Getting Started (New UW Students) 
* Create an account on Euler.
* [Download MobaXterm](https://mobaxterm.mobatek.net/) if on Windows. MacOS has built in ssh client.
* Clone repo on your euler account.
* Complete [GPU Usage on Euler](https://docs.google.com/presentation/d/1RmMtwF6Z7PBDQQaiICZhlcQqWpHeisHBgzWjIXjlAYA/edit?usp=sharing) tutorial.
* If using mscdata node [read this too!](https://docs.google.com/presentation/d/1vzh9ySl76F0Tl92PmUWIGG095do95-cHOgoayfLBjVM/edit?usp=sharing)
* Setup conda env.
* add conda env to kernel.
  * source activate myenv
  * python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
* Read papers on our [Mendeley](https://www.mendeley.com/?interaction_required=true) group. 
* Replicate results on FreeSolv dataset.
  ### Background Lingo
  * Crash course on [SMILES String](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) notation
  * Get started with [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
  * What's a [CAS Registry Number?](https://en.wikipedia.org/wiki/CAS_Registry_Number)
  * Benchmark data to replicate results from most research papers at [MoleculeNet](http://moleculenet.ai/).
  * Our method for converting IUPAC -> SMILES. [OPSIN Tool.](https://opsin.ch.cam.ac.uk/)
  * Convert  molecular images -> SMILES. [OSRA Tool.](https://cactus.nci.nih.gov/osra/)

## Contributing Data
We've web scraped chemical data from chemical supplier websites. 
To commit new data please follow checklist bellow. It'll change with trial and error.

- [ ] Format dataframe as below.

| SMILES :smiley:            |              Compound                 | Cas No        | FlashPoint (Celsius)  |
| :-------------------------:|:-------------------------------------:|:-----------:  | :--------------------:|
| C(C)(=O)NCCC[Si](OC)(OC)OC | (3-ACETAMIDOPROPYL)TRIMETHOXYSILANE   | 57757-66-1   | 35 |

- [ ] Valide SMILES strings with RDKit.
- [ ] Remove duplicate SMILES strings.
- [ ] Commit new data with Last Name, First Name, Data Source, Website, Web Pages in title.

## Credits
* ChemNet KDD 2018 [paper.](https://www.kdd.org/kdd2018/accepted-papers/view/using-rule-based-labels-for-weak-supervised-learning-a-chemnet-for-transfer)

## Support
<p float="left">
<img src="http://wacc.wisc.edu/assets/images/sbel_logo.png" alt="alt text" width="300" height="100">
<img src="http://wacc.wisc.edu/assets/images/skunkworks.png" alt="alt text" width="150" height="150">
 <img src="https://skunkworks.engr.wisc.edu/wp-content/uploads/sites/712/2016/03/MRSEC-UWM.jpg" alt="alt text" width="300" height="100">
</p>
