# Benchmark of the Experimental campaign for Tin Transport in Asdex

# 1. Extract exerimental results from aug_sfutils exploiting Aurora package 
# 2. Compute Prad and Nparticles

# 1. Call Aurora and take here only the results 
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import numpy as np
import fileinput
import subprocess
import json

import aug_sfutils as sf 
sys.path.append("/shares/departments/AUG/users/ebray/a8run")
import AuroraRun
from AuroraRun import AuroraCall
import PreparingShotFile
from PreparingShotFile import ExperimentalFiles
import PostProcessing
from PostProcessing import ErrorBar

plt.ion()
#def main(): 

# let's try with the json file 
inputFile = open('input.json') 
input = json.load(inputFile)

shot = input['Aurora']['shot']  
diagnostic = input['Aurora']['diagnostic']
timeShot = input['Aurora']['timeShot']
Impurity = input['Aurora']['Impurity']
SourceType = input['Aurora']['SourceType']
SourceRate = input['Aurora']['SourceRate']
TeovTi = input['Aurora']['TeovTi']
Zeff = input['Aurora']['Zeff']

baseDir = input['trview']['baseDir'] 
shotFileNumber = input['trview']['shotFileNumber'] 
expFileName = input['trview']['expFileName'] 
equFileName = input['trview']['equFileName']

OutputFileName = input['PostProc']['OutputFileName']  

Pradiated,Nparticles,nz,nz_sep = AuroraRun.AuroraCall(shot, diagnostic, timeShot, Impurity, SourceType, SourceRate, TeovTi, Zeff, Plot=True)
print("Prad BLB [MW]: ", Pradiated)
print("Npart: ", Nparticles)
print("nz at rho = 0.85: {:e}".format(nz_sep))

'''
PostProcessing.KinProfiles(baseDir,OutputFileName)
PradASTRA = PostProcessing.ImpProfile(baseDir,OutputFileName)
PostProcessing.ErrorBar(baseDir,OutputFileName,shot,timeShot) 


print("Prad ASTRA [MW] :", PradASTRA)
'''
'''
baseDir = input['trview']['baseDir']  
shotFileNumber = input['trview']['shotFileNumber']  
expFileName = input['trview']['expFileName'] 
equFileName = input['trview']['equFileName'] 


# When working this is the line to just call the routine and check it worked  
CheckCopiedFiles = PreparingShotFile.ExperimentalFiles(baseDir, shotFileNumber, expFileName, equFileName)


# (3. Run trview for the experimental files to give to ASTRA - to be done externally)
# 4. Copy and modify files produced by trview in the right repository 

baseDir = '/shares/departments/AUG/users/ebray'
shotFileNumber = '41277' 
expFileName = 'aug41277'
equFileName = 'TinCoeff'



# When working this is the line to just call the routine and check it worked  
#CheckCopiedFiles = PreparingShotFile.ExperimentalFiles(baseDir, shotFileNumber, expFileName, equFileName)


# copy the remaining file saved in None into the shares directory of my user 
# Per far questo devo essere connesso in toki 
bashCommand_toki = 'ssh -Y ebray@toki01.bc.rzg.mpg.de' 
bashCommand_cpFromNone = 'cp udb/'+ shotFileNumber + '/*_AVG0 ' + baseDir + '/udb/' + shotFileNumber
cpOfCoords = 'cp udb/' + shotFileNumber + '/' + shotFileNumber + '.bnd* ' + baseDir + '/udb/' + shotFileNumber

# copy repository #shot created into the shares to the udb repository of my folder    
bashCommand_cp = 'cp -r ' + baseDir + '/udb/' + shotFileNumber + ' ' + baseDir + '/a8/udb/'

# copy exp, nml and nbi file from None to a8 
cpOfexp = 'cp exp/' + expFileName + ' ' + baseDir + '/exp'
cpOfnml = 'cp exp/nml' + expFileName + ' ' + baseDir + '/exp/nml'
cpOfnbi = 'cp exp/nbi' + expFileName + ' ' + baseDir + '/exp/nbi'

# modify the exp and nml file to use the _AVG0 files 
# now put yourself in your directory 
exitFromssh = 'exit'
cdMydir = 'betta' 
for line in fileinput.input('/exp/' + expFileName, inplace=True):
    if 'IPL ' in line:
        print('IPL      U-file:41277/MAG41277.IPL_AVG0', end='\n')
    if 'BTOR ' in line:
        print('BTOR      U-file:41277/MAG41277.BTOR_AVG0', end='\n')
    if 'TE ' in line:
        print('TE       U-file:41277/TE41277.IDA_AVG0:1.e-3', end='\n')
    if 'TI ' in line:
        print('TI       U-file:41277/TI41277.CUZ_AVG0:1.e-3', end='\n')
    if 'NE ' in line:
        print('NE       U-file:41277/NE41277.IDA_AVG0:1.e-13', end='\n')   
    if 'CAR1 ' in line:
        print('CAR1     U-file:41277/ANGF41277.CUZ_AVG0:1.', end='\n')   


# change in the exp/nml file
for line in fileinput.input('/exp/nml' + expFileName, inplace=True):
    if '   pecr_file  = ' in line:
        print('   pecr_file  = \'udb/41277/P41277.ECH_AVG0\' ', end='\n')
    if '   theta_file = ' in line:
        print('   theta_file = \'udb/41277/THEAS41277.ECH_AVG0\'', end='\n')
    if '   phi_file   = ' in line:
        print('   phi_file   = \'udb/41277/PHIAS41277.ECH_AVG0\'', end='\n')
    if '   pinj_file =' in line:
        print('   pinj_file = \'udb/41277/P41277.NBI_AVG0\'', end='\n')


# Should decrease the number of points for the Q profile (safetyfactor)
# !!!!! per ora forse faccio prima a farlo per conto mio !!!! 

#  Run the simulation 
runSim =  'exe/as_exe -m ' + equFileName + ' -v ' + expFileName + ' -dev aug -s 0. -e 10. -tpause 10.' 


# 5. Run the iterative ASTRA method 
#   a. fixed kinetic profiles and impurity evolution 
#       i. do it changing the boundary condition of F1 to match the radiated power from Aurora  
#   b. fixed impurity and evolution of bck plasma --> matching of experimental profiles 
# (6. IMEP)  
# 7. Comparison with the SOLPS-ITER output at the separatrix (averaging strategy)
#   a. Impurity source 
#   b. Transport coefficients  
'''


