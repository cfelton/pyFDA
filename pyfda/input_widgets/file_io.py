# -*- coding: utf-8 -*-
"""
Widget for exporting / importing and saving / loading filter data

Author: Christian Muenker
"""
from __future__ import print_function, division, unicode_literals, absolute_import
import sys, os, io
import logging
logger = logging.getLogger(__name__)

from ..compat import QtCore, QFD, Qt, QWidget, QPushButton, QFont, QFrame, QVBoxLayout

import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

try:
    import xlwt
except ImportError:
    XLWT = False
    logger.info("Module xlwt not installed -> no *.xls coefficient export")
else:
    XLWT = True

try:
    import XlsxWriter as xlsx
except ImportError:
    XLSX = False
    logger.info("Module XlsxWriter not installed -> no *.xlsx coefficient export")
else:
    XLSX = True

#try:
#    import xlrd
#except ImportError:
#    XLRD = False
#    logger.info("Module xlrd not installed -> no *.xls coefficient import")
#else:
#    XLRD = True


import pyfda.filterbroker as fb # importing filterbroker initializes all its globals
import pyfda.pyfda_rc as rc
from pyfda.pyfda_io_lib import extract_file_ext

class File_IO(QWidget):
    """
    Create the widget for saving / loading data
    """

    sigFilterLoaded = QtCore.pyqtSignal() # emitted when filter has been loaded successfully

    def __init__(self, parent):
        super(File_IO, self).__init__(parent)

        self._construct_UI()

    def _construct_UI(self):
        """
        Intitialize the user interface
        -
        """
        # widget / subwindow for parameter selection
        self.butSave = QPushButton("Save Filter", self)
        self.butLoad = QPushButton("Load Filter", self)


        # ============== UI Layout =====================================
        bfont = QFont()
        bfont.setBold(True)

        bifont = QFont()
        bifont.setBold(True)
        bifont.setItalic(True)

        ifont = QFont()
        ifont.setItalic(True)

        layVIO = QVBoxLayout()
        layVIO.addWidget(self.butSave) # save filter dict -> various formats
        layVIO.addWidget(self.butLoad) # load filter dict -> various formats

        # This is the top level widget, encompassing the other widgets
        frmMain = QFrame(self)
        frmMain.setLayout(layVIO)

        layVMain = QVBoxLayout()
        layVMain.setAlignment(Qt.AlignTop)
#        layVMain.addLayout(layVIO)
        layVMain.addWidget(frmMain)
        layVMain.setContentsMargins(*rc.params['wdg_margins'])


        self.setLayout(layVMain)

        #----------------------------------------------------------------------
        # SIGNALS & SLOTs
        #----------------------------------------------------------------------
        self.butSave.clicked.connect(self.save_filter)
        self.butLoad.clicked.connect(self.load_filter)

#------------------------------------------------------------------------------
    def load_filter(self):
        """
        Load filter from zipped binary numpy array or (c)pickled object to
        filter dictionary and update input and plot widgets
        """
        file_filters = ("Zipped Binary Numpy Array (*.npz);;Pickled (*.pkl)")
        dlg = QFD(self)
        file_name, file_type = dlg.getOpenFileName_(
                caption = "Load filter ", directory = rc.save_dir,
                filter = file_filters)
        file_name = str(file_name) # QString -> str

        for t in extract_file_ext(file_filters): # get a list of file extensions
            if t in str(file_type):
                file_type = t

        if file_name != "": # cancelled file operation returns empty string

            # strip extension from returned file name (if any) + append file type:
            file_name = os.path.splitext(file_name)[0] + file_type

            file_type_err = False
            try:
                with io.open(file_name, 'rb') as f:
                    if file_type == '.npz':
                        # http://stackoverflow.com/questions/22661764/storing-a-dict-with-np-savez-gives-unexpected-result
                        a = np.load(f) # array containing dict, dtype 'object'

                        for key in a:
                            if np.ndim(a[key]) == 0:
                                # scalar objects may be extracted with the item() method
                                fb.fil[0][key] = a[key].item()
                            else:
                                # array objects are converted to list first
                                fb.fil[0][key] = a[key].tolist()
                    elif file_type == '.pkl':
                        if sys.version_info[0] < 3:
                            fb.fil = pickle.load(f)
                        else:
                        # this only works for python >= 3.3
                            fb.fil = pickle.load(f, fix_imports = True, encoding = 'bytes')
                    else:
                        logger.error('Unknown file type "%s"', file_type)
                        file_type_err = True
                    if not file_type_err:
                        logger.info('Loaded filter "%s"', file_name)
                         # emit signal -> InputTabWidgets.load_all:
                        self.sigFilterLoaded.emit()
                        rc.save_dir = os.path.dirname(file_name)
            except IOError as e:
                logger.error("Failed loading %s!\n%s", file_name, e)
            except Exception as e:
                logger.error("Unexpected error:", e)

#------------------------------------------------------------------------------
    def file_dump (self, fOut):
        """
        Dump file out in custom text format that apply tool can read to know filter coef's
        """

#       Fixed format widths for integers and doubles
        intw = '10'
        dblW = 27
        frcW = 20

#       Fill up character string with filter output
        filtStr = '# IIR filter\n'

#       parameters that made filter (choose smallest eps)
#       Amp is stored in Volts (linear units)
#       the second amp terms aren't really used (for ellip filters)

        FA_PB  = fb.fil[0]['A_PB']
        FA_SB  = fb.fil[0]['A_SB']
        FAmp = min(FA_PB, FA_SB)

#       Freq terms in radians so move from -1:1 to -pi:pi
        f_lim = fb.fil[0]['freqSpecsRange']
        f_unit = fb.fil[0]['freq_specs_unit']

        F_S   = fb.fil[0]['f_S']
        if fb.fil[0]['freq_specs_unit'] == 'f_S':
            F_S = F_S*2
        F_SB  = fb.fil[0]['F_SB'] * F_S * np.pi
        F_SB2 = fb.fil[0]['F_SB2'] * F_S * np.pi
        F_PB  = fb.fil[0]['F_PB'] * F_S * np.pi
        F_PB2 = fb.fil[0]['F_PB2'] * F_S * np.pi

#       Determine pass/stop bands depending on filter response type
        passMin = []
        passMax = []
        stopMin = []
        stopMax = []

        if fb.fil[0]['rt'] == 'LP':
            passMin = [ -F_PB,      0,    0]
            passMax = [  F_PB,      0,    0]
            stopMin = [-np.pi,   F_SB,    0]
            stopMax = [ -F_SB,  np.pi,    0]
            f1 = F_PB
            f2 = F_SB
            f3 = f4 = 0
            Ftype = 1
            Fname = 'Low_Pass'

        if fb.fil[0]['rt'] == 'HP':
            passMin = [-np.pi,   F_PB,    0]
            passMax = [ -F_PB,  np.pi,    0]
            stopMin = [ -F_SB,      0,    0]
            stopMax = [  F_SB,      0,    0]
            f1 = F_SB
            f2 = F_PB
            f3 = f4 = 0
            Ftype = 2
            Fname = 'Hi_Pass'

        if fb.fil[0]['rt'] == 'BS':
            passMin = [-np.pi,  -F_PB, F_PB2]
            passMax = [-F_PB2,   F_PB, np.pi]
            stopMin = [-F_SB2,   F_SB,     0]
            stopMax = [ -F_SB,  F_SB2,     0]
            f1 = F_PB
            f2 = F_SB
            f3 = F_SB2
            f4 = F_PB2
            Ftype = 4
            Fname = 'Band_Stop'

        if fb.fil[0]['rt'] == 'BP':
            passMin = [-F_PB2,   F_PB,     0]
            passMax = [ -F_PB,  F_PB2,     0]
            stopMin = [-np.pi,  -F_SB, F_SB2]
            stopMax = [-F_SB2,   F_SB, np.pi]
            f1 = F_SB
            f2 = F_PB
            f3 = F_PB2
            f4 = F_SB2
            Ftype = 3
            Fname = 'Band_Pass'

        filtStr = filtStr + '{:{align}{width}}'.format('10',align='>',width=intw)+ ' IIRFILT_4SYM\n'
        filtStr = filtStr + '{:{align}{width}}'.format(str(Ftype),align='>',width=intw)+ ' ' + Fname + '\n'
        filtStr = filtStr + '{:{d}.{p}f}'.format(FAmp,d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(passMin[0],d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(passMax[0],d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(passMin[1],d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(passMax[1],d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(passMin[2],d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(passMax[2],d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(stopMin[0],d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(stopMax[0],d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(stopMin[1],d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(stopMax[1],d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(stopMin[2],d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(stopMax[2],d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(f1,d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(f2,d=dblW,p=frcW) + '\n'
        filtStr = filtStr + '{: {d}.{p}f}'.format(f3,d=dblW,p=frcW)
        filtStr = filtStr + '{: {d}.{p}f}'.format(f4,d=dblW,p=frcW) + '\n'

#       move pol/res/gain into terms we need
        Fdc  = fb.fil[0]['rpk'][2]
        rC   = fb.fil[0]['rpk'][0]
        pC   = fb.fil[0]['rpk'][1]
        Fnum = len(pC)

#       Gain term
        filtStr = filtStr + '{: {d}.{p}e}'.format(Fdc,d=dblW,p=frcW) + '\n'

#       Real pole count inside the unit circle (none of these)

        filtStr = filtStr + '{:{align}{width}}'.format(str(0),align='>',width=intw) + '\n'

#       Complex pole/res count inside the unit circle

        filtStr = filtStr + '{:{i}d}'.format(Fnum, i=intw)+ '\n'

#       Now dump poles/residues
        for j in range(Fnum):
            filtStr = filtStr + '{:{i}d}'.format(j, i=intw) + ' '
            filtStr = filtStr + '{: {d}.{p}e}'.format(rC[j].real,d=dblW,p=frcW) + ' '
            filtStr = filtStr + '{: {d}.{p}e}'.format(rC[j].imag,d=dblW,p=frcW) + ' '
            filtStr = filtStr + '{: {d}.{p}e}'.format(pC[j].real,d=dblW,p=frcW) + ' '
            filtStr = filtStr + '{: {d}.{p}e}'.format(pC[j].imag,d=dblW,p=frcW) + '\n'

#       Real pole count outside the unit circle (none of these)
        filtStr = filtStr + '{:{align}{width}}'.format(str(0),align='>',width=intw) + '\n'

#       Complex pole count outside the unit circle (none of these)
        filtStr = filtStr + '{:{align}{width}}'.format(str(0),align='>',width=intw) + '\n'

#       Now write huge text string to file
        fOut.write(filtStr)

#------------------------------------------------------------------------------

    def save_filter(self):
        """
        Save filter as zipped binary numpy array or pickle object
        """
        file_filters = ("Zipped Binary Numpy Array (*.npz);;Pickled (*.pkl);;Text file pole/residue (*.txt)")
        dlg = QFD(self)
        # return selected file name (with or without extension) and filter (Linux: full text)
        file_name, file_type = dlg.getSaveFileName_(
                caption = "Save filter as", directory = rc.save_dir,
                filter = file_filters)

        file_name = str(file_name)  # QString -> str() needed for Python 2.x
        # Qt5 has QFileDialog.mimeTypeFilters(), but under Qt4 the mime type cannot
        # be extracted reproducibly across file systems, so it is done manually:

        for t in extract_file_ext(file_filters): # get a list of file extensions
            if t in str(file_type):
                file_type = t           # return the last matching extension

        if file_name != "": # cancelled file operation returns empty string

            # strip extension from returned file name (if any) + append file type:
            file_name = os.path.splitext(file_name)[0] + file_type

            file_type_err = False
            try:
                if file_type == '.txt':
                    # save as a custom residue/pole text output for apply with custom tool
                    # make sure we have the residues
                    if 'rpk' in fb.fil[0]:
                        with io.open(file_name, 'w', encoding="utf8") as f:
                            self.file_dump(f)
                    else:
                        file_type_err = True
                        logger.error('filter has no residues/poles, cannot save txt file')
                else:
                    with io.open(file_name, 'wb') as f:
                        if file_type == '.npz':
                            np.savez(f, **fb.fil[0])
                        elif file_type == '.pkl':
                            # save as a pickle version compatible with Python 2.x
                            pickle.dump(fb.fil, f, protocol = 2)
                        else:
                            file_type_err = True
                            logger.error('Unknown file type "%s"', file_type)

                if not file_type_err:
                    logger.info('Filter saved as "%s"', file_name)
                    rc.save_dir = os.path.dirname(file_name) # save new dir

            except IOError as e:
                logger.error('Failed saving "%s"!\n%s\n', file_name, e)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    from ..compat import QApplication
    app = QApplication(sys.argv)
    mainw = File_IO(None)

    app.setActiveWindow(mainw)
    mainw.show()

    sys.exit(app.exec_())
