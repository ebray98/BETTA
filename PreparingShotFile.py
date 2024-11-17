import sys
import os
import fileinput
import subprocess

def ExperimentalFiles(baseDir, shotFileNumber, expFileName, equFileName):

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


    output = 'All files copied!'

    return output