# AI-Pediatric Brain Tumor

## preprocess

### run rename_the_folder.py

**Ser2Mod.xlsx**:
save the convert from series description to modility

**rename_the_folder.py**: 
change the folder name to the series description of the dcm file it contains

**form_dict_and_modility_normlize**:
normlize the series description to modility and load the Ser2Mod.xlsx to form a dict as {'series description' : 'modility'} 

**remove_redunant_file_and_dir**:
delete useless dir and file

