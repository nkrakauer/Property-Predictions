## Property-Predictions :zap::battery::zap:
Neural Network that predicts properties for new materials.

## Getting Started (UW Students) 
* Create an account on Euler.
* [Download MobaXterm](https://mobaxterm.mobatek.net/) if on Windows. MacOS has built in ssh client.
* Clone repo on your euler account.
* Complete [GPU Uage on Euler](https://docs.google.com/presentation/d/1RmMtwF6Z7PBDQQaiICZhlcQqWpHeisHBgzWjIXjlAYA/edit?usp=sharing) tutorial.
* If using mscdata node [read this too!](https://docs.google.com/presentation/d/1vzh9ySl76F0Tl92PmUWIGG095do95-cHOgoayfLBjVM/edit?usp=sharing)
* Setup Conda env. 
* Read papers on our [Mendeley](https://www.mendeley.com/?interaction_required=true) group. 
  ### Background Lingo
  * Crash course on [SMILES String](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) notation
  * Get started with [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html)

## Contributing Data
When you're given a PDF of the data you need to get "creative". We've web scraped chemical data from chemical supplier websites.
To commit new data please follow checklist bellow. It'll change with trial and error, I promise :smirk:.

- [ ] Format dataframe as below

| SMILES :smiley:            |              Compound                 | Cas No        | FlashPoint (Celsius)  |
| :-------------------------:|:-------------------------------------:|:-----------:  | :--------------------:|
| C(C)(=O)NCCC[Si](OC)(OC)OC | (3-ACETAMIDOPROPYL)TRIMETHOXYSILANE   | 57757-66-1   | 35 |

- [ ] Valide SMILES strings with RDKit.
- [ ] Removed duplicate SMILES strings.
- [ ] Commit new data with Last Name, First Name, Data Source, Website, Web Pages in title

## Support
<p float="left">
<img src="http://wacc.wisc.edu/assets/images/sbel_logo.png" alt="alt text" width="300" height="100">
<img src="http://wacc.wisc.edu/assets/images/skunkworks.png" alt="alt text" width="100" height="100">
</p>
