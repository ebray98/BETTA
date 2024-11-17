# In this module I'm extracting the value of Prad considering it is defined by the bolometers at a specific 
# time during the simulation. Moreover I'm multplying for the *0.286 which is the parameters which should make 
# the computations return compatible values.. It should be substituted with a routine to pass from the 
# line of sights of the bolometers to the radial coordinate  

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from omfit_classes import omfit_eqdsk
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import time

import aug_sfutils as sf 

# the "diagnostic", "Impurity" parameter has to be passed as a string inside "" 
# The SourceType and SourceRate refers to the impurity  

def AuroraCall(shot, diagnostic, timeShot, Impurity, SourceType, SourceRate, TeovTi, Zeff, Plot=False):

    plt.ion()
    # load experimental data and compute Prad from bolometers 
    
    sys.path.append("../")
    import aurora 

    blb = sf.SFREAD(shot, diagnostic)

    timeFVC = blb.gettimebase("powFVC")
    powerFVC = blb.getobject("powFVC")
    TotPowFVC = np.nansum(powerFVC, axis=1) # faccio la somma di tutte le los available nel tempo  
    
    timeFHC = blb.gettimebase("powFHC")
    powerFHC = blb.getobject("powFHC") 
    TotPowFHC = np.nansum(powerFHC, axis=1) # faccio la somma di tutte le los available nel tempo  

    TotPow = TotPowFVC + TotPowFHC

    # the time base of these BLB diagnostic is something strange 
    # let's take the time base of the IDA diagnostic that is the one referring to the "physical" time of the shot 
    
    ida = sf.SFREAD(shot, "IDA")
    Time = ida.gettimebase("Te")    
    
    '''
    spred = sf.SFREAD(shot, "SCL", ed=5)
    Time = spred.gettimebase("Sn39_138")
    '''
    PowerInterp = np.interp(Time, timeFVC, TotPow*0.286)

    idx_Prad = np.abs(Time - timeShot).argmin()
    # in this case it was needed to subtract the radiated power in background - if not just uncomment the first Prad  
    timeBck = 2.6 
    idx_Bck = np.abs(Time - timeBck).argmin()  
    # Choose the first Prad if you want to know the power only radiated by Tin  
    #Prad = (PowerInterp[idx_Prad] - PowerInterp[idx_Bck])*1e-6
    # Choose the second Prad if you want to know the radiated power of the shot  
    Prad = PowerInterp[idx_Bck]*1e-6
    print("PradBLB: ", Prad)
    # --------------------------------------------------------------------------------------------------------------- #  

    # Compute number of particles of impurity  
    # aurora & FACIT setup --> this is needed to set a proper boundary condition for the impurity in astra  

    # set to 0 in case of light impurity 
    rotation_model = 2

    # initial guesses 
    D_an = 1.e4 # [cm^2/s]  --> 1 m^2/s
    V_an = -1e2 # [cm/s]    --> -1 m/s

    
    geqdsk = OMFITgeqdsk("").from_aug_sfutils(shot= shot, time= timeShot, eq_shotfile="EQI")
    
    namelist =  aurora.default_nml.load_default_namelist()
    kp = namelist["kin_profs"] 

    ida = sf.SFREAD(shot, "ida")
    time_ida = ida.gettimebase("Te") # time base of the temperature profile 
    it_ida = np.argmin(np.abs(time_ida-timeShot)) # returns the index of the time "time" in the array extracted from the aug data
    rhop_ida = ida.getareabase("Te") # Reads the areabase  
    Te_eV = ida.getobject("Te")
    ne_m3 = ida.getobject("ne")


    # assign the extract data to the namelist   
    rhop_kp = kp["Te"]["rhop"] = kp["ne"]["rhop"] = rhop_ida[:, it_ida] 
    kp["Te"]["vals"] = Te_eV[:,it_ida] # eV
    kp["ne"]["vals"] = ne_m3[:,it_ida] * 1e-6 # from m^-3 --> to cm^-3

    # set impurity species and sources rate
    imp = namelist["imp"] = Impurity 
    namelist["source_type"] = SourceType
    namelist["source_rate"] = SourceRate #part/s  --> take from SOLPS or from ITERATIVE process to match Prad


    # Starting from the initial guess condition of the "source rate", I impose a loop to try to match the computed
    # radiated power delivered by Tin (mainly) 

    oldSource = namelist["source_rate"]
    f = 0
    Ptarget = Prad*1e6
    err = 100
    tol = 1e-2
    Nztot = np.zeros(320) #320 is the dimension of the final grid (roa variable)  
    PradTot = 1.7e6 # initial guess 

    while err > tol: 

        namelist["source_rate"] = oldSource*(1+f)

        # Aurora setup 
        asim =  aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

        times_DV = np.array([0])
        nz_init = np.zeros((asim.rvol_grid.size, asim.Z_imp + 1))

        # initialize transport coefficients
        D_z = np.zeros((asim.rvol_grid.size, times_DV.size, asim.Z_imp + 1))  # space, time, nZ
        V_z = np.zeros(D_z.shape)

        # set time-independent anomalous transport coefficients
        Dz_an = np.zeros(D_z.shape)  # space, time, nZ
        Vz_an = np.zeros(D_z.shape)

        # set anomalous transport coefficients
        Vz_an[:] = V_an
        Dz_an[:] = D_an

        # -------------------
        # prepare FACIT input
        rr = asim.rvol_grid / 100  # in m
        idxsep = np.argmin(np.abs(1.0 - asim.rhop_grid))  # index of radial position of separatrix 
        amin = rr[idxsep]  # minor radius in m
        roa = rr[: idxsep + 1] / amin  # normalized radial coordinate 

        B0 = np.abs(geqdsk["BCENTR"])  # magnetic field on axis
        R0 = geqdsk["fluxSurfaces"]["R0"]  # major radius

        qmag = np.interp(roa, geqdsk["RHOVN"], -geqdsk["QPSI"])[: idxsep + 1]  # safety factor
        rhop = asim.rhop_grid[: idxsep + 1]

        # profiles
        Ni = (np.interp(roa, rhop_kp, kp["ne"]["vals"]) * 1e6)  # in m**3 instead of cm**3 in FACIT 
        TeovTi =  TeovTi # 2.0 # kp["Te"]["vals"]/kp["Ti"]["vals"] # electron to ion temperature ratio
        Ti = np.interp(roa, rhop_kp, kp["Te"]["vals"]) / TeovTi  # I verified that is almost half of Te from #41278  
        Te = np.interp(roa, rhop_kp, kp["Te"]["vals"])
        Ne = np.interp(roa, rhop_kp, kp["ne"]["vals"])

        gradNi = np.gradient(Ni, roa*amin)
        gradTi = np.gradient(Ti, roa*amin)

        gradTi[-1] = gradTi[-2]
        gradNi[-1] = gradNi[-2]


        # !!!!! For now consider the case of Tin, but if another impurity is considered this has to be changed !!!!  
        def ZTIN(Temp): 
                
            T = np.arange(1, 10001)
            Z = np.log10(T.astype(float))
            Y = []


            for temperature in T:
                    if temperature < 100:
                        Y.append(0.)
                    elif 100 <= temperature < 300:
                        Y.append(-1.74287118071950e-01 * Z[temperature - 1]**3 +
                                1.54636650598728e+00 * Z[temperature - 1]**2 -
                                3.99141529108946e+00 * Z[temperature - 1]**1 +
                                4.33147417838939e+00 * Z[temperature - 1]**0)
                    elif 300 <= temperature < 1000:
                        Y.append(1.00244293446694e+00 * Z[temperature - 1]**3 -
                                7.97732921344918e+00 * Z[temperature - 1]**2 +
                                2.13382972994841e+01 * Z[temperature - 1]**1 -
                                1.78615834244534e+01 * Z[temperature - 1]**0)
                    elif 1000 <= temperature < 2000:
                        Y.append(3.42895052030529e-01 * Z[temperature - 1]**3 -
                                3.06822566369654e+00 * Z[temperature - 1]**2 +
                                9.53786318906057e+00 * Z[temperature - 1]**1 -
                                8.83692882480517e+00 * Z[temperature - 1]**0)
                    elif 2000 <= temperature < 5000:
                        Y.append(4.81585016923541e-01 * Z[temperature - 1]**3 -
                                5.25915388459379e+00 * Z[temperature - 1]**2 +
                                1.92606216460337e+01 * Z[temperature - 1]**1 -
                                2.20499427661916e+01 * Z[temperature - 1]**0)
                    elif 5000 <= temperature < 10000:
                        Y.append(-2.08321206186342e+00 * Z[temperature - 1]**3 +
                                2.39727274395118e+01 * Z[temperature - 1]**2 -
                                9.17468909033947e+01 * Z[temperature - 1]**1 +
                                1.18408176981176e+02 * Z[temperature - 1]**0)
                    else:
                        Y.append(9.91829921918504e-02 * Z[temperature - 1]**3 -
                                1.32853805480940e+00 * Z[temperature - 1]**2 +
                                5.94848074638099e+00 * Z[temperature - 1]**1 -
                                7.22498252575176e+00 * Z[temperature - 1]**0)

            ZITIN = 10**np.array(Y)

            ZTin_Interp = np.interp(Temp, T, ZITIN)
            return ZTin_Interp


        def ZWOL(Temp):

            TEL = np.arange(0.1, 100)  # Inserisci i valori dell'array TE qui
            ZIWOL_PUET2010 = np.zeros_like(TEL, dtype=float)  # Inizializza ZIWOL_PUET2010 con zero

            for j, te in enumerate(TEL):
                T = te * 1000.0
                Z = np.log10(te)

                if T <= 9.86:
                    ZIWOL_PUET2010[j] = 0.0
                elif 9.86 < T <= 1299.99:
                    ZIWOL_PUET2010[j] = 13.3595 * Z + 26.803
                elif 1299.99 < T <= 3043.6871:
                    ZIWOL_PUET2010[j] = 38.996 * Z + 23.882
                elif 3043.6871 < T <= 8749.84:
                    ZIWOL_PUET2010[j] = 19.8975 * Z + 33.1165
                elif T > 8749.84:
                    ZIWOL_PUET2010[j] = 20.0115 * (Z - 0.942)**0.6773 + 51.86

            ZIWOL = ZIWOL_PUET2010

            ZWol_Interp = np.interp(Temp, TEL, ZIWOL)
            return ZWol_Interp

        # TO CHECK Look if Zeff is working - it must update once aurora has computed nz_init  
        #ZIMP = ZTIN(Te)
        ZIMP = ZWOL(Te)
        Zeff = Zeff * np.ones(roa.size) # 1.5 * np.ones(roa.size)
        
        # ----------------------------------------------------------------------------------------------------------- # 
        # This part on the Machi can be changed depending on the presence of an experimental profile or other  
        # computation of Machi from a typical H-mode heavy impurity case (Paper Daniel)

        import csv 
        file = open('plot_data.csv')
        csvreader = csv.reader(file)
        header = [] 
        header = next(csvreader)
        rows = []

        for row in csvreader: 
            rows.append(row)

        file.close()

        gridCSV = [] 
        Mz_starRough = [] 
        for sottolista in rows: 
            primoValore = float(sottolista[0])
            secondoValore = float(sottolista[1])
            gridCSV.append(primoValore)
            Mz_starRough.append(secondoValore)

        Mz_star = np.interp(roa, gridCSV, Mz_starRough)

        # ----------------------------------------------------------------------------------------------------------- # 

        # uncomment to begin simulation from a pre-existing profile
        c_imp = 1e-4 # trace concentration
        for k in range(nz_init.shape[1]):
            nz_init[:idxsep+1,k] = c_imp*Ni*1e-6 # in 1/cm**3
        
        
        if rotation_model == 0:

            Machi = np.zeros(
                roa.size
            )  # no rotation (not that it matters with rotation_model=0)
            RV = None
            ZV = None

        elif rotation_model == 2:

            Machi = np.sqrt(Mz_star**2/(asim.A_imp/asim.main_ion_A - ZIMP/asim.main_ion_Z * Zeff/(Zeff+TeovTi)))

            nth = 51
            theta = np.linspace(0, 2 * np.pi, nth)

            RV, ZV = aurora.rhoTheta2RZ(geqdsk, rhop, theta, coord_in="rhop", n_line=201)
            RV, ZV = RV.T, ZV.T

        else:
            raise ValueError("Other options of rotation_model are not enabled in this example!")

        # ----------
        # call FACIT

        starttime = time.time()
        for j, tj in enumerate(times_DV):

            for i, zi in enumerate(range(asim.Z_imp + 1)):

                if zi != 0:
                    Nz = nz_init[: idxsep + 1, i] * 1e6  # in 1/m**3
                    gradNz = np.gradient(Nz, roa * amin)

                    fct = aurora.FACIT(
                        roa,
                        zi,
                        asim.A_imp,
                        asim.main_ion_Z,
                        asim.main_ion_A,
                        Ti,
                        Ni,
                        Nz,
                        Machi,
                        Zeff,
                        gradTi,
                        gradNi,
                        gradNz,
                        amin / R0,
                        B0,
                        R0,
                        qmag,
                        rotation_model=rotation_model,
                        Te_Ti=TeovTi,
                        RV=RV,
                        ZV=ZV,
                    )

                    D_z[: idxsep + 1, j, i] = fct.Dz * 100**2  # convert to cm**2/s
                    V_z[: idxsep + 1, j, i] = fct.Vconv * 100  # convert to cm/s

        time_exec = time.time() - starttime
        print("FACIT exec time [s]: ", time_exec)

        # add anomalous transport
        D_z += Dz_an
        V_z += Vz_an

        
        # correction diffusion coefficients  
        target_value = 0.05
        diff = np.abs(asim.rhop_grid - target_value)
        indCorrection = np.argmin(diff)
        D_z[:indCorrection]  = D_z[indCorrection] 
        V_z[:indCorrection]  = V_z[indCorrection] 
        

        # run Aurora forward model
        out = asim.run_aurora(D_z, V_z, times_DV=times_DV, nz_init=None, plot=False)

        # extract densities and particle numbers in each simulation reservoir
        nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
        
        sum_curve = np.sum(nz*1e6, axis=1)
        #print("sum_curve: ", sum_curve)
        #Nztot = np.interp(roa, asim.rvol_grid, sum_curve[:,-1])
        #print(f"NztotAfter: ", Nztot[-1]) # part/s 

        #I try to compute the radiated power  !!! POI AGGIUNGI CX E NEUTRALS !!! 
        asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te, Ti=Ti, 
        prad_flag=True, spectral_brem_flag=False)
        PradTot = aurora.grids_utils.vol_int(asim.rad['tot'].transpose(1,0)[:,-1],asim.rvol_grid,asim.pro_grid, asim.Raxis_cm)
        # I computed the average values of the transport coefficients K and H to reproduce the ones in the paper

        print("PradTot: ", PradTot)
        oldSource = namelist["source_rate"]

        f = (Ptarget-PradTot)/Ptarget
        err = abs(f)

        #print(f'PradTot: %d [MW] ', PradTot*1e-6)
        #print(f'err:', err)
        #print(f'f:', f)
        #print(f'Source: ', namelist["source_rate"])

    Npart = aurora.grids_utils.vol_int(nz.transpose(2,1,0)[-1,:,:], asim.rvol_grid, asim.pro_grid, asim.Raxis_cm)
    Npart = np.sum(Npart)

    # POTREI ANCHE FARMI FARE DUE CONTI PER OTTENERE LA BC ESATTA PER ASTRA  
    if Plot == True:
        nzSum = np.sum(nz[:,:,-1]*1e6, axis=1)
        sep = 0.85
        index = np.argmin(np.abs(asim.rhop_grid - sep))
        nz_sep =  nzSum[index]

        plt.figure()
        for i in range(len(nz[1])):
            plt.plot(asim.rhop_grid, nz[:,i,-1]*1e6)

        plt.plot(asim.rhop_grid, nzSum,color = 'k', label="sum")
        plt.grid(True)
        plt.legend()
        plt.show()


    return Prad, Npart, nz, nz_sep