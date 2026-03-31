# Capillary Analysis Process

Take videos and/or static frames of capillary bridges
put onto external harddrive
convert videos into frames with Analysis_Files/utilities/convert_video.py
upload all frames (each experiment in its own subfolder) to the HPC using winSCP client
go to OOD.HPC.arizona.edu and start a GPU-enabled VsCode session on Ocelote (4 cores, 2 hours)
run minimal_test.py with arrow at top right to make sure environment is properly loaded.
if so, run Batch_capillary_multi_folder.py (or something like that), select all folders with images, then hit cancel in tkinter window and it will process SAM masks for every frame
You can close your vsCode tab and it will still run on the HPC
come back after an hour or two and you can use winSCP client to download the mask files to your external harddrive
Do all analysis from analysis.ipynb
1. select image folder, mask folder, output folder (create output as needed)
2. define substrate lines (select relevant frames, define lines, the code interpolates between selected frames)
3. verify the substrate lines are good, and check that classifier works for dataset. retrain classifier if needed using: Analysis_Files\Classifier_files\labelling_script.py -> Analysis_Files\Classifier_files\LeftRight_labeling_script.py -> Analysis_Files\Classifier_files\Train_Mask_Classifier.py
4. run plot_forces_multiprocessing to fit ellipses to contours and calculate capillary forces. 
5. view force vs. separation graph (colored by expansions/contractions) 