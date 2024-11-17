import numpy as np
import matplotlib.pyplot as plt
import aug_sfutils as sf 
from scipy.io import netcdf as nc 
import pandas as pd 
import matplotlib.ticker as ticker


def KinProfiles(baseDir,OutputFileName):

    #plt.close('all')
    plt.ion()

    # import ASTRA results from ncdf files  
    astra_dat = nc.netcdf_file(baseDir + '/a8/MyOutput/' + OutputFileName, mmap=False).variables
    
    plt.style.use("default")
    xx = astra_dat['XRHO'].data
    n = -1

    # Background plasma parameters (Te, ne, ni & transport coefficients)


    fig,axs = plt.subplots()
    ax1 = axs.twinx()
    axs.plot(xx,astra_dat['TE'].data[n],c = 'firebrick',  label = "$T_{e}$", linewidth=2.5)
    axs.plot(xx,astra_dat['TEX'].data[n], c = 'firebrick', ls = '--', label = "$T_{e,exp}$", linewidth=2.5)
    axs.plot(xx,astra_dat['TI'].data[n], c = 'b',  label = "$T_{i}$", linewidth=2.5)
    axs.plot(xx,astra_dat['TIX'].data[n],c = 'b', ls = '--', label = "$T_{i,exp}$", linewidth=2.5)
    ax1.plot(xx,astra_dat['NE'].data[n], c = 'g', label = "$n_{e}$", linewidth=2.5)
    ax1.plot(xx,astra_dat['NEX'].data[n], c = 'g', ls = '--', label = "$n_{e,exp}$", linewidth=2.5)
    ax1.set_title("Reference case")
    axs.set_ylabel("T [keV]")
    ax1.set_ylabel("ne $[m^{-3}$]", color = 'g')
    axs.set_xlabel(r"$\rho_p$")
    axs.grid(True)
    axs.legend(loc = (0.8,0.5))
    ax1.legend(loc = (0.8, 0.8))
    #axs.legend(loc = 'best')
    plt.tight_layout()


    # car58 è XIMAIN 
    plt.figure()
    plt.plot(xx,astra_dat['XI'].data[n], label = "$Total$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR21'].data[n], label = "$Turbolent$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR11'].data[n], ls = '-.', label = "$Neoclassical$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR58'].data[n], ls = '-.', label = "$XIMAIN$", linewidth=2.5)
    plt.title("Ion heat conductivity", fontsize = 16)
    plt.tick_params(axis='y', labelsize=14) 
    plt.tick_params(axis='x', labelsize=14) 
    plt.ylabel("[$m^2/s$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)

    # car59 è HEMAIN  
    plt.figure()
    plt.plot(xx,astra_dat['HE'].data[n], label = "$Total$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR22'].data[n], label = "$Turbolent$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR12'].data[n], ls = '-.', label = "$Neoclassical$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR59'].data[n], ls = '-.', label = "$HEMAIN$", linewidth=2.5)
    plt.title("Electron heat conductivity", fontsize = 16)
    plt.tick_params(axis='y', labelsize=14) 
    plt.tick_params(axis='x', labelsize=14) 
    plt.ylabel("[$m^2/s$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)

    #print(astra_dat['CAR12'].data[n])


    plt.figure()
    plt.plot(xx,astra_dat['DN'].data[n], label = "$Total$", linewidth=2.5)
    #plt.plot(xx,astra_dat['CAR23'].data[n], ls = '--', label = "$Turbolent$", linewidth=2.5)
    plt.title("Electron diffusion coefficient - only neo", fontsize = 16)
    plt.tick_params(axis='y', labelsize=14) 
    plt.tick_params(axis='x', labelsize=14) 
    plt.ylabel("[$m^2/s$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)

    # car61 è CNMAIN
    plt.figure()
    plt.plot(xx,astra_dat['CN'].data[n], label = "$Total$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR24'].data[n], ls = '--', label = "$Turbolent$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR15'].data[n], ls = '-.', label = "$Neoclassical$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR61'].data[n], ls = '-.', label = "$CNMAIN$", linewidth=2.5)
    plt.title("Electron convective coefficient", fontsize = 16)
    plt.tick_params(axis='y', labelsize=14) 
    plt.tick_params(axis='x', labelsize=14) 
    plt.ylabel("[$m/s$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)


def ImpProfile(baseDir,OutputFileName): 
    
    # Impurity distribution, transport coefficients and impurity radiation 
    #plt.close('all')
    plt.ion()

    # import ASTRA results from ncdf files  
    astra_dat = nc.netcdf_file(baseDir + '/a8/MyOutput/' + OutputFileName, mmap=False).variables
    
    plt.style.use("default")
    xx = astra_dat['XRHO'].data
    n = -1

    plt.figure()
    plt.plot(xx,astra_dat['NIZ1'].data[n]*1e19, c = 'k', label = "$N_z$", linewidth=2.5)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.tick_params(axis='y', labelsize=14)  # Modifica '14' alla dimensione del font desiderata


    plt.title("Impurity distribution", fontsize = 16)
    plt.ylabel(r"$N_z$ $[part/m^3]$", fontsize = 16)
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)
    plt.savefig("Nz.png") 

    '''
    prtin = astra_dat['CAR60'].data[n]/(astra_dat['NE'].data[n]*astra_dat['NIZ1'].data[n])

    plt.figure()
    plt.plot(xx,astra_dat['PRAD'].data[n],c = 'b',  label = "$P_{rad}$", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR60'].data[n],c = 'r',  label = "$P_{tin}$", linewidth=2.5)
    plt.title("Power density")
    plt.ylabel("[$MW/m^{3}$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)
    '''

    plt.figure()
    plt.plot(xx,astra_dat['CAR30'].data[n],c = 'g',  label = "Tot", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR54'].data[n],c = 'royalblue', ls = '--',  label = "Neocl", linewidth=2.5)
    plt.tick_params(axis='y', labelsize=14)  # Modifica '14' alla dimensione del font desiderata
    plt.title("$D_{imp}$", fontsize = 16)
    plt.ylabel("[$m/s^2$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(xx,astra_dat['CAR31'].data[n],c = 'green',  label = "Tot", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR55'].data[n],c = 'royalblue', ls = '--', label = "Neocl", linewidth=2.5)
    plt.plot(xx,astra_dat['CAR50'].data[n],c = 'darkorange', ls = '-.', label = "Turbolent", linewidth=2.5)
    plt.tick_params(axis='y', labelsize=14)  # Modifica '14' alla dimensione del font desiderata
    plt.title("$V_{imp}$", fontsize = 16)
    plt.ylabel("[$m/s$]")
    plt.xlabel(r'$\rho$')
    plt.legend()
    plt.grid(True)

    PradASTRA = astra_dat['CAR8'].data[n][-1] 
    
    return PradASTRA




def ErrorBar(baseDir,OutputFileName,shot,timeShot):
  
    #plt.close('all')
    plt.ion()

    # import ASTRA results from ncdf files  
    astra_dat = nc.netcdf_file(baseDir + '/a8/MyOutput/' + OutputFileName, mmap=False).variables
    #astra_dat = nc.netcdf_file('/shares/departments/AUG/users/ebray/a8/MyOutput/BckFixedImp1_4.CDF', mmap=False).variables

    plt.style.use("default")

    xx = astra_dat['XRHO'].data
    n = -1

    ne_exp = astra_dat['NEX'].data[n]
    ne_sim = astra_dat['NE'].data[n]

    Te_exp = astra_dat['TEX'].data[n]*1e3
    Te_sim = astra_dat['TE'].data[n]*1e3

    Ti_exp = astra_dat['TIX'].data[n]*1e3
    Ti_sim = astra_dat['TI'].data[n]*1e3


    # import aug_sfutils and experimental uncertainty data   
    shot = shot
    time_shot = timeShot

    ida = sf.SFREAD(shot, "ida")
    list_objects = ida.getlist()
    time_ida = ida.gettimebase("ne_unc") # time base of the temperature profile 
    it_ida = np.argmin(np.abs(time_ida-time_shot)) # returns the index of the time "time" in the array extracted from the aug data
    rhop_ida = ida.getareabase("ne_unc") # Reads the areabase 
    rhop = rhop_ida[:,it_ida]  

    # uncertainty on ne  
    ne_unc = ida.getobject("ne_unc")
    ne = ida.getobject("ne")

    # uncertainty on Te  
    Te_unc = ida.getobject("Te_unc")
    Te = ida.getobject("Te")


    # Ne error  
    ne_upper = (ne[:,it_ida]) + ne_unc[:,it_ida] 
    ne_lower = (ne[:,it_ida]) - ne_unc[:,it_ida]

    plt.figure()

    plt.fill_between(rhop, ne_lower, ne_upper, color='lightskyblue', alpha=0.5, label='Uncertainty', linewidth = 2.0)
    plt.plot(rhop, ne[:,it_ida] , ls = '--', label = "$n_{e,exp}$", linewidth = 2.0)
    plt.plot(xx,ne_sim*1e19, label = "$n_{e,sim}$", linewidth = 2.0)

    plt.title("Ne error bar", fontsize = 14)
    plt.xlabel(r'$\rho$', fontsize = 13)
    plt.ylabel("ne $[m^{-3}$]", fontsize = 13)
    plt.ylim(0, 1.5e20)
    plt.xlim(0,1)
    plt.tick_params(axis='both', labelsize = 12)
    plt.legend(fontsize = 14)
    plt.grid(True)
    plt.show()

    # Te error 
    Te_upper = (Te[:,it_ida]) + Te_unc[:,it_ida] 
    Te_lower = (Te[:,it_ida]) - Te_unc[:,it_ida]

    plt.figure()

    #plt.errorbar(rhop, Te_exp_interp, Te_unc[:,it_ida], fmt='o')

    plt.fill_between(rhop, Te_lower, Te_upper, color='lightskyblue', alpha=0.5, label='Uncertainty', linewidth = 2.0)
    plt.plot(rhop,Te[:,it_ida] , ls = '--', label = "$T_{e,exp}$", linewidth = 2.0)
    plt.plot(xx,Te_sim, label = "$T_{e,sim}$", linewidth = 2.0)

    plt.title("Te error bar", fontsize = 14)
    plt.xlabel(r'$\rho$', fontsize = 13)
    plt.ylabel("Te[keV]", fontsize = 13)
    plt.ylim(0,5300)
    plt.xlim(0,1)
    plt.tick_params(axis='both', labelsize = 12)
    plt.legend(fontsize = 14)
    plt.grid(True)
    plt.show() 



    # Ti cuz is referenced on the LOS - I try to apply the regression from the LOS to the rho 

    cuz = sf.SFREAD(shot, "cuz")
    R = cuz.getobject("R")
    z = cuz.getobject("z")
    phi = cuz.getobject("phi")
    Ti = cuz.getobject("Ti_c")
    LOS = cuz.getareabase("Ti_c")

    time_cuz = cuz.gettimebase("Ti_c")
    it_cuz = np.argmin(np.abs(time_cuz-time_shot))
    Ti = Ti[it_cuz,:] 
    Ti_unc = cuz.getobject("err_Ti_c")
    Ti_unc = Ti_unc[it_cuz,:] 
    rhop_cuz = cuz.getareabase("Ti_c")
    maxInd = np.argmax(Ti)

    TiToPlot = Ti[:maxInd-1]
    Ti_UncToPlot = Ti_unc[:maxInd-1] 
    rhoToPlot = rhop_cuz[:maxInd-1,it_cuz]

    # smoothing 

    min_value = np.min(rhoToPlot)
    max_value = np.max(rhoToPlot)
    norm_rho = (rhoToPlot - min_value) / (max_value - min_value)

    # Ti error 
    Ti_upper = (TiToPlot[::-1]  + Ti_unc[:maxInd-1]) 
    Ti_lower = (TiToPlot[::-1]  - Ti_unc[:maxInd-1])


    plt.figure()
    plt.plot(norm_rho[::-1] , TiToPlot[::-1], ls = '--', label = '$T_{i,exp}$')
    plt.fill_between(norm_rho[::-1], Ti_lower, Ti_upper, color='lightskyblue', alpha=0.5, label='Uncertainty', linewidth = 2.0)
    plt.plot(xx, Ti_sim, label = "$T_{i,sim}$", linewidth = 2.0)

    plt.title("Ti error bar", fontsize = 14)
    plt.xlabel(r'$\rho$', fontsize = 13)
    plt.ylabel("Ti[keV]", fontsize = 13)
    plt.tick_params(axis='both', labelsize = 12)
    plt.legend(fontsize = 14)
    plt.grid(True)
    plt.show() 
    
