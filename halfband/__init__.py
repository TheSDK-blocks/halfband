# halfband class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 05.01.2018 15:17
import os
import sys
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time
#Add TheSDK to path. Importing it first adds the rest of the modules
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))
from thesdk import *

from refptr import *
from rtl import *

#Simple buffer template
class halfband(rtl,thesdk):
    def __init__(self,*arg): 
        self.proplist = [ 'Rs' ];    #properties that can be propagated from parent
        self.Rs = 160;                 # sampling frequency
        self.halfband_Bandwidth=0.45 # Pass band bandwidth
        self.halfband_N=40           #Number of coeffs
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self._Z = refptr();
        self._classfile=os.path.dirname(os.path.realpath(__file__)) + "/"+__name__
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()

    def init(self):
        self.H=self.firhalfband(**{'n':self.halfband_N, 'bandwidth':self.halfband_Bandwidth})
        self.def_rtl()
        rndpart=os.path.basename(tempfile.mkstemp()[1])
        self._infile=self._rtlsimpath +'/A_' + rndpart +'.txt'
        self._outfile=self._rtlsimpath +'/Z_' + rndpart +'.txt'
        self._rtlcmd=self.get_rtlcmd()

    def get_rtlcmd(self):
        #the could be gathered to rtl class in some way but they are now here for clarity
        submission = ' bsub -K '  
        rtllibcmd =  'vlib ' +  self._workpath + ' && sleep 2'
        rtllibmapcmd = 'vmap work ' + self._workpath

        if (self.model is 'vhdl'):
            rtlcompcmd = ( 'vcom ' + self._rtlsrcpath + '/' + self._name + '.vhd '
                          + self._rtlsrcpath + '/tb_'+ self._name+ '.vhd' )
            rtlsimcmd =  ( 'vsim -64 -batch -t 1ps -g g_infile=' + 
                           self._infile + ' -g g_outfile=' + self._outfile 
                           + ' work.tb_' + self._name + ' -do "run -all; quit -f;"')
            rtlcmd =  submission + rtllibcmd  +  ' && ' + rtllibmapcmd + ' && ' + rtlcompcmd +  ' && ' + rtlsimcmd

        elif (self.model is 'sv'):
            rtlcompcmd = ( 'vlog -work work ' + self._rtlsrcpath + '/' + self._name + '.sv '
                           + self._rtlsrcpath + '/tb_' + self._name +'.sv')
            rtlsimcmd = ( 'vsim -64 -batch -t 1ps -voptargs=+acc -g g_infile=' + self._infile
                          + ' -g g_outfile=' + self._outfile + ' work.tb_' + self._name  + ' -do "run -all; quit;"')

            rtlcmd =  submission + rtllibcmd  +  ' && ' + rtllibmapcmd + ' && ' + rtlcompcmd +  ' && ' + rtlsimcmd

        else:
            rtlcmd=[]
        return rtlcmd

    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            self.par=False

        if self.model=='py':
            fid=open(self._infile,'wb')
            np.savetxt(fid,self.iptr_A.Value.reshape(-1,1).view(float),fmt='%i', delimiter='\t')
            fid.close()
            self.decimate_input()
        else: 
          try:
              os.remove(self._infile)
          except:
              pass
          fid=open(self._infile,'wb')
          np.savetxt(fid,self.iptr_A.Value.reshape(-1,1).view(float),fmt='%i', delimiter='\t')
          fid.close()
          while not os.path.isfile(self._infile):
              self.print_log({'type':'I', 'msg':"Wait infile to appear"})
              time.sleep(5)
          try:
              os.remove(self._outfile)
          except:
              pass
          self.print_log({'type':'I', 'msg':"Running external command %s\n" %(self._rtlcmd) })
          subprocess.call(shlex.split(self._rtlcmd));
          
          while not os.path.isfile(self._outfile):
              self.print_log({'type':'I', 'msg':"Wait outfile to appear"})
              time.sleep(5)
          fid=open(self._outfile,'r')
          out = np.loadtxt(fid,dtype=complex)
          #Of course it does not work symmetrically with savetxt
          out=(out[:,0]+1j*out[:,1]).reshape(-1,1) 
          fid.close()
          if self.par:
              queue.put(out)
          self._Z.Value=out
          os.remove(self._infile)
          os.remove(self._outfile)

    def decimate_input(self):
            if self.iptr_A.Value.shape[1] > self.iptr_A.Value.shape[0]:
                out=np.convolve(np.transpose(self.iptr_A.Value)[:,0],self.H[:,0]).rehape(-1,1)
            else:
                print(self.H.shape)
                out=np.convolve(self.iptr_A.Value[:,0],self.H[:,0]).reshape(-1,1)

            out=out[0::2,0]
            if self.par:
                queue.put(out)
            self._Z.Value=out

    def firhalfband(self,**kwargs):
       n=kwargs.get('n',32)
       if np.remainder(n,2) > 0:
           self.print_log({'type':'F', 'msg':'Number of coefficients must be even'})
       bandwidth=kwargs.get('bandwidth',0.45) # Fs=1
       desired=np.array([ 1, 0] )
       bands=np.array([0, bandwidth, 0.499,0.5])
       coeffs=sig.remez(n, bands, desired, Hz=1)
       hb=np.zeros((2*n-1,1))
       hb[0::2,0]=coeffs
       hb[n-1,0]=1
       return hb
   
    def export_scala(self):
       #This is the first effort to support generators
       bwstr=str(self.halfband_Bandwidth).replace('.','')
       print(bwstr)
       packagestr="halfband_BW_"+bwstr+"_N_"+str(self.halfband_N)
       tapfile=os.path.dirname(os.path.realpath(__file__)) +"/"+packagestr+".scala"
       fid=open(tapfile,'w')
       msg="object "+ packagestr+" {\n"
       fid.write(msg)
       msg="val H=Seq("
       fid.write(msg)
       lines=self.H.shape[0]
       for k in range(lines-1):
           fid.write("%0.32f,\n" %(self.H[k]))
       fid.write("%0.32f)\n }\n" %(self.H[lines-1]))
       fid.close()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  halfband import *
    from  f2_signal_gen import *
    from  f2_system import *
    siggen=f2_signal_gen()
    siggen.bbsigdict={ 'mode':'sinusoid', 'freqs':[11.0e6 , 0.45*80e6, 0.95*80e6, 1.05*80e6 ], 'length':2**14, 'BBRs':160e6 };
    siggen.Users=1
    siggen.Txantennas=1
    siggen.init()
    #Mimic ADC
    bits=10
    insig=siggen._Z.Value[0,:,0].reshape(-1,1)
    insig=np.round(insig/np.amax(np.abs(insig))*(2**(bits-1)-1))
    print(insig)
    h=halfband()
    h.iptr_A.Value=insig
    h.halfband_Bandwidth=0.45
    h.halfband_N=40
    h.init()
    impulse=np.r_['0', h.H, np.zeros((1024-h.H.shape[0],1))]
    h.export_scala() 
    h.run() 

    w=np.arange(1024)/1024
    spe1=np.fft.fft(impulse,axis=0)
    f=plt.figure(1)
    plt.plot(w,20*np.log10(np.abs(spe1)/np.amax(np.abs(spe1))))
    plt.ylim((-80,3))
    plt.grid(True)
    f.show()

    nbits=16
    spe2=np.fft.fft(np.round(impulse*(2**(nbits-1)-1)),axis=0)
    g=plt.figure(2)
    plt.plot(w,20*np.log10(np.abs(spe2)/np.amax(np.abs(spe2))))
    plt.ylim((-80,3))
    plt.grid(True)
    g.show()
    
    #spe3=np.fft.fft(h._Z.Value,axis=0)
    fs, spe3=sig.welch(h._Z.Value,fs=80e6,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
    w=np.arange(spe3.shape[0])/spe3.shape[0]
    ff=plt.figure(3)
    plt.plot(w,10*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
    plt.ylim((-80,3))
    plt.grid(True)
    ff.show()

    #Required to keep the figures open
    input()

