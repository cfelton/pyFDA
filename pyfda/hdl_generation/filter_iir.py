# -*- coding: utf-8 -*-
#
# Copyright (c) 2011, 2015 Christopher L. Felton
#

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

"""
IIR Hadward Filter Generation
=============================

The following is a straight forward HDL description of a Direct Form I IIR
filter and an object that encapsulates the design and configuration of the
IIR filter.  This module can be used to generate synthesizable Verilog/VHDL
for ASIC of FPGA implementations.

How to use this module
-----------------------

   >>> flt = FilterIIR()

This code is discussed in the following
http://www.fpgarelated.com/showarticle/7.php
http://dsp.stackexchange.com/questions/1605/designing-butterworth-filter-in-matlab-and-obtaining-filter-a-b-coefficients-a


:Author: Christopher Felton <cfelton@ieee.org>
"""

from myhdl import (toVerilog, toVHDL, Signal, ResetSignal, always, delay,
                   instance, instances, intbv, traceSignals,
                   Simulation, StopSimulation)

import numpy as np
from numpy import pi, log10
from numpy.fft import fft
from numpy.random import uniform
from scipy import signal

from .filter_intf import FilterInterface
from .filter_iir_hdl import filter_iir_hdl, filter_iir_sos_hdl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FilterIIR(object):
    def __init__(self, b=None, a=None, sos=None, word_format=(24,0), sample_rate=1):
        """
        In general this object generates the HDL based on a set of
        coefficients or a sum of sections matrix.  This object can
        only be used for 2nd order type=I IIR filters or cascaded 2nd orders.
        The coefficients are passed as floating-point values are and converted
        to fixed-point using the format defined in the `W` argument.  The
        `W` argument is a tuple that defines the overall word length (`wl`),
        the interger word length (`iwl`) and the fractional word length (`fwl`):
        `W= (wl, iwl, fwl,)`.

        Arguments
        ---------
          b: the filters transfer function numerator coefficients
          a: the filters transfer function denominator coefficients
          sos: the sum-of-sections array coefficients, if used `a` and `b`
            are ignored
          W: word format (fixed-point) used

        @todo: utilize the myhdl.fixbv type for fixed point format (`W`)
        """
        # The W format, intended to be (total bits, integer bits,
        # fraction bits) is not fully supported.
        # Determine the max and min for the word-widths specified
        w = word_format
        self.word_format = w
        self.max = int(2**(w[0]-1))
        self.min = int(-1*self.max)
        self._sample_rate = sample_rate
        self._clock_frequncy = 20

        # properties that are set/configured
        self._sample_rate = 1
        self._clock_frequncy = 20
        self._shared_multiplier = False

        # create the fixed-point (integer) version of the coefficients
        self._convert_coefficients(a, b, sos)

    def _convert_coefficients(self, a, b, sos):
        """ Extract the coefficients and convert to fixed-point (integer)

        Arguments
        ---------
        :param a:
        :param b:
        :param sos:

        """
        # @todo: use myhdl.fixbv, currently the fixed-point coefficient
        # @todo: the current implementation only uses an "all fractional"
        # @todo: format.
        if len(self.word_format) == 2 and self.word_format[1] != 0:
            raise NotImplementedError
        elif (len(self.word_format) == 3 and
              self.word_format[1] != 0 and
              self.word_format[2] != self.word_format[1]-1):
            raise NotImplementedError

        # @todo: if a sum-of-sections array is supplied only the first
        # @todo: section is currently used to compute the frequency
        # @todo: response, the full response is needed.
        N, Wn = 2, 0
        self.is_sos = False
        if sos is not None:
            self.is_sos = True
            (self.n_section, o) = sos.shape
            self.b = sos[:, 0:3]
            self.a = sos[:, 3:6]
        else:
            self.b = b
            self.a = a
            self.n_section = 1

        # fixed-point Coefficients for the IIR filter
        # @todo: use myhdl.fixbv, see the comments and checks above
        # @todo: the fixed-point conversion is currently very limited.
        self.fxb = np.round(self.b * self.max)/self.max
        self.fxa = np.floor(self.a * self.max)/self.max

        # Save off the frequency response for the fixed-point
        # coefficients.
        if not self.is_sos:
            self.w, self.h = signal.freqz(self.fxb, self.fxa)
            self.hz = (self.w/(2*pi) * self._sample_rate)
        else:
            self.w = [None for _ in range(self.n_section)]
            self.h = [None for _ in range(self.n_section)]
            for ii in range(self.n_section):
                self.w[ii], self.h[ii] = signal.freqz(self.fxb[ii], self.fxa[ii])
            self.w, self.h = np.array(self.w), np.array(self.h)
            self.hz = (self.w/(2*pi) * self._sample_rate)

        # Create the integer version
        if self.is_sos:
            self.fxb = self.fxb * self.max
            self.fxa = self.fxa * self.max
        else:
            self.fxb = tuple(map(int, self.fxb*self.max))
            self.fxa = tuple(map(int, self.fxa*self.max))

        # used by the RTL simulation to generate freq response
        self.yfavg, self.xfavg, self.pfavg = None, None, None

        # golden model - direct-form I IIR filter using floating-point
        # coefficients
        self.iir_sections = [None for _ in range(self.n_section)]

        # @todo: determine the section used type 1, 2, 3, 4 ...
        # @todo: _iir_section = _iir_type_one_section if self.form_type == 1 else _iir_type_two_section
        if self.n_section > 1:
            for ii in range(self.n_section):
                self.iir_sections[ii] = _iir_type_one_section(self.b[ii], self.a[ii])
        else:
            self.iir_sections[0] = _iir_type_one_section(self.b, self.a)

        self.first_pass = True

    def filter_direct_form_one(self, x):
        """Floating-point IIR filter direct-form 1
        """
        for ii in range(self.n_section):
            x = self.iir_sections[ii].process(x)
        y = x

        return y

    def filter_direct_form_two(self, x):
        """Floating-point IIR filter direct-form 2
        """
        raise NotImplementedError("Directe Form II not implemented")
        for ii in range(self.n_sections):
            x = self.iir_sections[ii].process(x)
        y = x
        return y

    def get_hdl(self, clock, reset, sigin, sigout):
        if self.is_sos:
            hdl = filter_iir_sos_hdl(clock, reset, sigin, sigout,
                                     coefficients=(self.fxb, self.fxa),
                                     shared_multiplier=self._shared_multiplier)
        else:
            hdl = filter_iir_hdl(clock, reset, sigin, sigout,
                                 coefficients=(self.fxb, self.fxa),
                                 shared_multiplier=self._shared_multiplier)
        return hdl

    def convert(self):
        """Convert the HDL description to Verilog and VHDL.
        """
        w = self.word_format
        imax = 2**(w[0]-1)

        # small top-level wrapper
        def filter_iir_top(clock, reset, x, xdv, y, ydv):
            sigin = FilterInterface(word_format=(len(x), 0, len(y)-1))
            sigin.data, sigin.data_valid = x, xdv
            sigout = FilterInterface(word_format=(len(y), 0, len(y)-1))
            sigout.data, sigout.dave_valid = y, ydv

            if self.sos:
                iir = filter_iir_sos_hdl(clock, reset, sigin, sigout,
                                         coefficients=(self.fxb, self.fxa),
                                         shared_multiplier=self._shared_multiplier)
            else:
                iir = filter_iir_hdl(clock, reset, sigin, sigout,
                                     coefficients=(self.fxb, self.fxa),
                                     shared_multiplier=self._shared_multiplier)
            return iir

        clock = Signal(False)
        reset = ResetSignal(0, active=1, async=False)
        x = Signal(intbv(0, min=-imax, max=imax))
        y = Signal(intbv(0, min=-imax, max=imax))
        xdv, ydv = Signal(bool(0)), Signal(bool(0))

        toVerilog(filter_iir_top, clock, reset, x, xdv, y, ydv)
        toVHDL(filter_iir_top, clock, reset, x, xdv, y, ydv)

    def simulate_freqz(self, num_loops=3, Nfft=1024):
        """ simulate the discrete frequency response
        This function will invoke an HDL simulation and capture the
        inputs and outputs of the filter.  The response can be compared
        to the frequency response (signal.freqz) of the coefficients.
        """
        self.Nfft = Nfft
        w = self.word_format[0]
        clock = Signal(bool(0))
        reset = ResetSignal(0, active=1, async=False)
        sigin = FilterInterface(word_format=(24,0,0))
        sigout = FilterInterface(word_format=(24,0,0))
        xf = Signal(0.0)    # floating point version

        # determine the sample rate to clock frequency
        fs = self._sample_rate
        fc = self._clock_frequency
        fscnt_max = fc//fs
        cnt = Signal(fscnt_max)

        # get the hardware description to simulation
        tbdut = traceSignals(self.get_hdl, clock, reset, sigin, sigout)

        @always(delay(10))
        def tbclk():
            clock.next = not clock

        @always(clock.posedge)
        def tbdv():
            if cnt == 0:
                cnt.next = fscnt_max
                sigin.data_valid.next = True
            else:
                cnt.next -= 1
                sigin.data_valid.next = False

        @always(clock.posedge)
        def tbrandom():
            if sigin.data_valid:
                xi = uniform(-1, 1)
                sigin.data.next = int(self.max*xi)
                xf.next = xi

        @instance
        def tbstim():
            ysave = np.zeros(Nfft)
            xsave = np.zeros(Nfft)
            psave = np.zeros(Nfft)

            self.yfavg = np.zeros(Nfft)
            self.xfavg = np.zeros(Nfft)
            self.pfavg = np.zeros(Nfft)

            for ii in range(num_loops):
                for jj in range(Nfft):
                    yield sigin.data_valid.posedge
                    xsave[jj] = float(sigin.data)/self.max
                    yield sigout.data_valid.posedge
                    ysave[jj] = float(sigout.data)/self.max

                    psave[jj] = self.filter_directt_form_one(float(xf))
                    #psave[jj] = self.filter_direct_form_two(float(xf))

                self.yfavg += (np.abs(fft(ysave, Nfft)) / Nfft)
                self.xfavg += (np.abs(fft(xsave, Nfft)) / Nfft)
                self.pfavg += (np.abs(fft(psave, Nfft)) / Nfft)

            raise StopSimulation

        return instances()

    def plot_response(self, ax):
        # Plot the designed filter response
        if self.n_section == 1:
            ax.plot(self.w, 20*log10(np.abs(self.h)), 'm')
            fxw, fxh = freqz(self.fxb, self.fxa)
            ax.plot(fxw, 20*log10(np.abs(fxh)), 'y:')

        # plot the simulated response
        #  -- Fixed Point Sim --
        xa = 2*pi * np.arange(self.Nfft)/self.Nfft
        H = self.yfavg / self.xfavg
        ax.plot(xa, 20*log10(H), 'b' )
        #  -- Floating Point Sim --
        Hp = self.pfavg / self.xfavg
        ax.plot(xa, 20*log10(Hp), 'g' )

        #pylab.axis((0, pi, -40, 3))
        ax.ylabel('Magnitude dB');
        ax.xlabel('Frequency Normalized Radians')
        ax.legend(('Ideal', 'Quant. Coeff.',
                      'Fixed-P. Sim', 'Floating-P. Sim'))
        ax.savefig(fn+".png")
        ax.savefig(fn+".eps")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IIRTypeOneSection(object):
    def __init__(self, b, a):
        self.b, self.a = b, a

        self._fbd = [0. for _ in range(2)]
        self._ffd = [0. for _ in range(2)]

    def process(self, x):

        y = x*self.b[0] + \
            self._ffd[0]*self.b[1] + \
            self._ffd[1]*self.b[2] - \
            self._fbd[0]*self.a[1] - \
            self._fbd[1]*self.a[2]

        self._ffd[1] = self._ffd[0]
        self._ffd[0] = x

        self._fbd[1] = self._fbd[0]
        self._fbd[0] = y

        return y


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IIRTypeTwoSection(object):
    def __init__(self, b, a):
        self.b = b
        self.a = a

    def process(self, x):
        # @todo: finish type two ...
        y = 0
        return y


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    # Instantiate the filter and define the Signal
    W = (8,0)
    b = np.asarray([1,1,1])
#    b = [1,1,1]
    a = np.asarray([3, 0, 2])
    # need to be ndarrays, with type list / tuple the filter "explodes" 
    flt = SIIR(W = W, b = b, a = a)

#    clk = Signal(False)
#    ts  = Signal(False)
#    x   = Signal(intbv(0,min=-2**(W[0]-1), max=2**(W[0]-1)))
#    y   = Signal(intbv(0,min=-2**(W[0]-1), max=2**(W[0]-1)))
#
    # Setup the Testbench and run
    print ("Simulation")
    tb = flt.TestFreqResponse(Nloops=3, Nfft=1024)
    sim = Simulation(tb)
    print ("Run Simulation")
    sim.run()
    print ("Plot Response")
    flt.PlotResponse()

    flt.Convert()
    print("Finished!")