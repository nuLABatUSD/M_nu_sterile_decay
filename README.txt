Matter_Power_Plots.py contains all the functions necessary to generating the delaP/P plots. All functions in the .py have notes on how to use them and what exactly they do. The same functions are also found in Matter_Power_Plots.ipynb where they can be run to see how the diagnostics are looking (Neff for example).

Out of all the functions that are in the .py, the first one that needs to be ran is make_data. Its first 6 inputs are all filenames that you pick the name of. These are files that will be created and saved as a .npz. The next 3 inputs are masses in MeV. The last input is a mass file that contains an e,f array. This fucntions creates all the data needed for the next function to just graph everything. Note: this function may take a while to run depending on the lifetime in the mass file. 

The next function that needs to be ran is make_graphs which generates the deltaP/P plots. This function takes the name of the 6 files entered in the previous function. Mass file is NOT an input

To get diagnostics (Neff for example) other functions must be run. Those are either going to be v_masses_nontherm_alpha or v_masses_therm depending on what you are trying to do. Very detailed notes about these functions and their inputs are located in the .py. 

Current plots as of 8/17 can be found in Potential_Fix.ipynb
