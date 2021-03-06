# -*- coding: utf-8 -*-
"""
pyfda_rc.py

This file contains layout definitions for Qt and matplotlib widgets
A dark and a light theme can be selected via a constant but this more a demonstration
on how to set things than a finished layout yet.

Default parameters, paths etc. are also defined at the end of the file.

Importing pyfda_rc runs the module once, defining all module variables
which are global (similar to class variables).

See
http://stackoverflow.com/questions/13034496/using-global-variables-between-files-in-python
http://pymotw.com/2/articles/data_persistence.html
for information on passing/storing data between files

See
http://doc.qt.io/qt-4.8/stylesheet-examples.html
http://www.informit.com/articles/article.aspx?p=1405556
for qss styling

Author: Christian Muenker
"""

from __future__ import division, unicode_literals, absolute_import
from pyfda import qrc_resources # contains all icons
import logging
logger = logging.getLogger(__name__)

from .pyfda_lib import cmp_version, CRLF

# #############################################################################
# General layout settings
# #############################################################################
# Get all available system styles:
#from PyQt5.QtWidgets import QStyleFactory
#fróm PyQt4.QtGui import QStyleFactory
#print(QStyleFactory.keys())

THEME = 'light' # select 'dark' or 'light' theme or 'none' or use one of the
                    # system styles like 'windows':
# QT4 only:         motif, cleanlooks, CDE, plastique
# QT5 only:         fusion
# MS Windows only:  windowsxp, windowsvista
# Mac only:         macintosh

mpl_ms = 8 # base size for matplotlib markers
# Various parameters for calculation and plotting
params = {'N_FFT':  2048,   # number of FFT points for plot commands (freqz etc.)
          'FMT': '{:.3g}',  # format string for QLineEdit fields
          'CSV':    # format options and parameters for CSV-files and clipboard
                  {
                  'delimiter' : ',', # default delimiter
                  'lineterminator'  : CRLF, # OS-dependend line break from pyfda_lib
                  'orientation': 'auto', # 'auto', 'vert', 'horiz'# table orientation
                  'header': 'off', # 'auto', 'on', 'off'
                  'clipboard': True # source/target is QClipboard or file
                  }, 
          'FMT_ba': 4,      # number of digits for coefficient table
          'FMT_pz': 5,      # number of digits for Pole/Zero table
          'P_Marker': [mpl_ms, 'r'], # size and color for poles' marker
          'Z_Marker': [mpl_ms, 'b'], # size and color for zeros' marker
          'wdg_margins' : (2,1,2,0),  # R, T, L, B widget margins
          'mpl_hatch_border': {'linewidth':1.0, 'color':'blue', 'linestyle':'--'}     
          }
mpl_params_dark = {
            'mpl_hatch': {                          # hatched area for specs
                         'facecolor': 'none', 
                         'hatch'    : '/', 
                         'edgecolor': '#808080',   # same as figure.edgecolor
                         'lw'       : 0.0},        # no border around hatched area
                         
            'mpl_stimuli':{                         # style for stimulus signals
                          'mfc': 'w', 'mec' : 'w', # marker face + edge color
                          'ms': mpl_ms,            # marker size
                          'alpha': 0.25,           # transparency (marker + stem)
                          'markerfmt':'*',         # marker symbol
                          'lw':'2' }               # stem linewidth
                    }

#fill_params = {'facecolor':'none','hatch':'/', 'edgecolor':rcParams['figure.edgecolor'], 'lw':0.0}
mpl_params_light = {
            'mpl_hatch': {                         # hatched area for specs
                         'facecolor': 'none',
                         'hatch'    : '/', 
                         'edgecolor': '#808080',   # same as figure.edgecolor
                         'lw'       : 0.0},        # no border around hatched area

            'mpl_stimuli':{                         # style for stimulus signals
                          'mfc': 'k', 'mec' : 'k', # marker face + edge color
                          'ms': mpl_ms,            # marker size
                          'alpha': 0.25,           # transparency (marker + stem)
                          'markerfmt':'*',         # marker symbol 
                          'lw':'2' }              # stem linewidth
                    }

# Dictionary with translations between short method names and long names for
# response types - the long name can be changed as you like, but don't change 
# change the short name - it is used to construct the filter design method names
rt_names = {"LP":"Lowpass", "HP":"Highpass", "BP":"Bandpass",
            "BS":"Bandstop", "AP":"Allpass", "MB":"Multiband",
            "HIL":"Hilbert", "DIFF":"Differentiator"}
            
# Dictionary with translations between short method names and long names for
# response types
ft_names = {"IIR":"IIR", "FIR":"FIR"}

# Dictionary dm_names is created dynamically by FilterTreeBuilder and stored
# in filterbroker.py

# Default savedir for filter coefficients, filter dicts etc.
save_dir = "D:/Daten"

# #############################################################################
# Matplotlib layout settings
# #############################################################################

# common matplotlib widget settings
mpl_rc = {'lines.linewidth'           : 1.5,
          'lines.markersize'          : mpl_ms,         # markersize, in points
          'font.family'               : 'sans-serif',#'serif',
          'font.style'                : 'normal',
          'mathtext.fontset'          : 'stixsans',#'stix',
          'mathtext.fallback_to_cm'   : True,
          'mathtext.default'          : 'it',
          'font.size'                 : 12, 
          'legend.fontsize'           : 12, 
          'axes.labelsize'            : 12, 
          'axes.titlesize'            : 14, 
          'axes.linewidth'            : 1, # linewidth for coordinate system
          'axes.formatter.use_mathtext': True, # use mathtext for scientific notation.
          'grid.linestyle'            : ':',
          'grid.linewidth'            : 0.5,
          'xtick.direction'           : 'out',
          'ytick.direction'           : 'out',
          #'xtick.top'                 : False, 2.0 only
          'figure.figsize'            : (5,4),
          'figure.dpi'                : 100
            }

mpl_2 =  {'hatch.color'               : '#808080',
          'hatch.linewidth'           : 0.5
          }

if cmp_version('matplotlib', '2.0') >= 0:
    mpl_rc.update(mpl_2)

# dark theme for matplotlib widgets
mpl_rc_dark = {
            'axes.facecolor'    : 'black',
            'axes.labelcolor'   : 'white',
            'axes.edgecolor'    : 'white',
            'figure.facecolor'  : '#202020',
            'figure.edgecolor'  : '#808080', # also color for hatched specs for mpl < 2.0
            'savefig.facecolor' : 'black',
            'savefig.edgecolor' : 'black', 
            'xtick.color'       : 'white',
            'ytick.color'       : 'white',
            'text.color'        : 'white',
            'grid.color'        : '#CCCCCC'
            }

try:
    # cycler is used by default with matplotlib 1.5 upwards
    from cycler import cycler
    CYC = True
    mpl_rc_dark.update(
            {'axes.prop_cycle': cycler('color', ['r', 'g', 'c', 'm', 'y', 'w'])})
except ImportError:
    CYC = False
    mpl_rc_dark.update(
            {'axes.color_cycle': ['r', 'g', 'c', 'm', 'y', 'w']})

# light theme for matplotlib widgets
mpl_rc_light = {
            'axes.facecolor'    : 'white',
            'axes.labelcolor'   : 'black',
            'axes.edgecolor'    : 'black',
            'figure.facecolor'  : 'white',
            'figure.edgecolor'  : '#808080', # also color for hatched specs for mpl < 2.0
            'savefig.facecolor' : 'white',
            'savefig.edgecolor' : 'white', 
            'xtick.color'       : 'black',
            'ytick.color'       : 'black',
            'text.color'        : 'black',
            'grid.color'        : '#222222'
            }
if CYC:
    mpl_rc_light.update({'axes.prop_cycle': cycler('color', ['r', 'b', 'c', 'm', 'k'])})
else:
    mpl_rc_light.update({'axes.color_cycle': ['r', 'b', 'c', 'm', 'k']})    
            

# --------------------- Matplotlib Fonts --------------------------------------
import matplotlib.font_manager
afm_fonts = sorted({f.name for f in matplotlib.font_manager.fontManager.afmlist})
ttf_fonts = sorted({f.name for f in matplotlib.font_manager.fontManager.ttflist})

if 'DejaVu Sans' in ttf_fonts:
    logger.info("Using 'DejaVu Sans' font.")
    mpl_rc.update({
                   'mathtext.fontset' : 'custom',
                   'mathtext.rm' : 'DejaVu Sans',
                   'mathtext.it' : 'DejaVu Sans:italic',
                   'mathtext.bf' : 'DejaVu Sans:bold'
                  })
elif 'Bitstream Vera Sans' in ttf_fonts:
    logger.info("Using 'Bitstream Vera Sans' font.")
    mpl_rc.update({
                   'mathtext.fontset' : 'custom',
                   'mathtext.rm' : 'Bitstream Vera Sans',
                   'mathtext.it' : 'Bitstream Vera Sans:italic',
                   'mathtext.bf' : 'Bitstream Vera Sans:bold'
                  })
# else: use sans-serif and stix-sans

# set all text to Stix font
#matplotlib.rcParams['mathtext.fontset'] = 'stixsans'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'

# #############################################################################
# QWidget style sheets (QSS)
# #############################################################################
      
# .Qxxx{} only matches Qxxx, not its children
#  #mylabel Qxxx{} only matches Qxxx with object name #mylabel
#  Qxxx Qyyy{} only matches Qyyy that is a child of Qxxx
#  Qxxx > Qyyy only matches Qyyy that is a direct child of Qxxxx
#  Qxxx:mystate only matches Qyyy in state 'mystate' (e.g. disabled)

#--------------- 
# dark QSS theme
#---------------
qss_dark = """
    .QWidget{color:white; background-color: black } /* background of application */
    QFrame{color:white;}
    QTextEdit{color: white; background-color: #444444;}
    QCheckBox{
        border: none;  /* dummy, needed to force using non-system widget rendering */    
        color: white;
        }
    QCheckBox::indicator{
     /* setting any properties here removes all default styles ... */
       }

    
    QScrollArea{background-color: #222222}
    QScrollArea > QWidget > QWidget{background-color: #222222}

    .QTabWidget::pane{color: white; background-color: #555555;} /* background of tab content */

    QLineEdit{background: #222222;
                border-style: outset;
                border-width: 2px;
                border-color: darkgrey;
                color: white;
    }
    QLineEdit:disabled{background-color:darkgrey;}
   
    QPushButton{
         background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 white, stop: 0.5 lightgray, stop: 1.0 #C2C7CB);
         color: black;
                    }
    
    QTableView{alternate-background-color:#222222;
        background-color:#444444; gridline-color: white;}
    QHeaderView{background-color:#222222;}
    QHeaderView::section{background-color:#111111;}
    QTableWidget QTableCornerButton::section{background-color:#444444;}
    QHeaderView::section:checked{background-color:rgb(190,1,1);}

    QComboBox QListView {color:black}
    QMessageBox{background-color:#444444}
            """
# ---------------         
# light QSS theme
# ---------------
qss_light = """
    .QWidget, .QFrame{color:black; background-color: white;}
    
    QScrollArea{color:black; background-color:white;}
    QScrollArea > QWidget > QWidget{color:black; background-color: white;}
    
    QTextEdit{background-color: white;}
    
    QTableWidget{color:black; background-color:white;}
    
    .QTabWidget::pane{background-color: #F0F0F0;} /* background of tab content */
    
    QLineEdit{background: white;
                border-color: darkgrey;}
    QLineEdit:disabled{background-color:darkgrey;}
  
    QPushButton{
         background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 white, stop: 0.5 lightgray, stop: 1.0 #C2C7CB);
                    }    
    QPushButton:disabled{color:darkgrey; }

    QHeaderView::section{background-color:darkgray; color:white;}    
    QHeaderView::section:checked{background-color:rgb(190,1,1); color:white;}
    
    QMessageBox{background-color: #EEEEEE}
    """


# common layout settings for QTabWidget
qss_tab_bar = """
 .QTabWidget::pane { /* The tab _widget_ frame */ 
    
     border-top: 2px solid #C2C7CB; 
     /*border : 5px solid;
     padding: 0;
     margin:  0; */
 }

 QTabWidget::tab-bar {
     left: 0.3em; /* move bar to the right: hack to prevent truncation of labels (QTBUG-6905) */
     }

/* Style the TAB using the tab sub-control. Note that it reads QTabBar _not_ QTabWidget */
 QTabBar {  font-weight: bold; font-size:11pt; }
 QTabBar::tab{
     color:black;
     font-size:10pt;
     font-weight:bold;
     background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                        stop: 0 white, stop: 0.5 lightgray, stop: 1.0 #C2C7CB);
     border: 1px solid #C4C4C3; 
     border-bottom-color: #C2C7CB; /* same as the pane color */
     border-top-left-radius: 4px;
     border-top-right-radius: 4px;
     min-width: 2em;
     padding: 0.2em;
 }

 QTabBar::tab:selected, QTabBar::tab:hover {background:lightblue;}
 
 QTabBar::tab:selected {
     border-color: #9B9B9B;
     border-bottom-color: #444444; /* same as pane color */
 }

 QTabBar::tab:!selected {
     margin-top: 0.2em; /* make non-selected tabs look smaller */
 }

 /* make use of negative margins to produce overlapping selected tabs */
 QTabBar::tab:selected {
     /* expand/overlap to both sides by 0.2em */
     margin-left: -0.2em;
     margin-right: -0.2em;
 }
 
 QTabBar::tab:first{
    /* the first tab */
}

 QTabBar::tab:first:!selected {
    /* the first unselected tab */
 }
  
 QTabBar::tab:first:selected {
     margin-left: 0; /* the first selected tab has nothing to overlap with on the left */
 }
 
 QTabBar::tab:last:selected {
     margin-right: 0; /* the last selected tab has nothing to overlap with on the right */
 }

 QTabBar::tab:only-one {
     margin: 0; /* if there is only one tab, we don't want overlapping margins */
 }
"""

# Common qss settings for all themes
qss_common = """
                QWidget{font-size:10pt; font-family: Tahoma;}
                /* Frame with control elements of all plot widgets */
                #frmControls{
                    border-top: solid darkgrey;
                    border-width: 2px;
                    margin: 0;
                    padding: 0;
                    }

                /* Frame for input subwidgets */
                QTabWidget#input_tabs > QFrame QFrame,                
                QTabWidget#input_tabs QTextBrowser
                {
                    /* background-color: pink; */
                    border: solid darkgrey;
                    border-width: 1px 0 1px 0;
                    padding: 0;
                    margin: 1px 0 0 0;
                    }

                /* Frame in frame, e.g. for target specs, only border-top */
                QTabWidget#input_tabs > QFrame QFrame .QFrame
                {
                    /* background-color:lime; */
                    border: solid darkgrey;
                    border-width: 1px 0 0 0;
                    padding: 0;
                    margin: 0;
                    }

                /* Dynamic filter subwidget */
                #wdg_fil{
                    /*background-color:lightblue;*/
                    border: none;
                    padding: 5px 0 0 0;
                    }

                /* setFrameStyle(QFrame.StyledPanel|QFrame.Sunken) */
                
                QPushButton
                {
                width: 20px;
                height: 20px;
                }

                QPushButton[state="normal"]{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 white, stop: 0.5 lightgray, stop: 1.0 #C2C7CB);
                                color: black;}
                                 
                QPushButton[state="changed"]{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #cccccc, stop: 0.1 yellow, stop: 1.0 #999999);
                                color: black;}

                QPushButton[state="error"]{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #cccccc, stop: 0.1 red, stop: 1.0 #444444);
                                color: white;}
                QPushButton[state="failed"]{background-color:orange; color:white}
                QPushButton[state="ok"]{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #cccccc, stop: 0.1 green, stop: 1.0 #444444);
                                color: white;}
                *[state="unused"], *[state="u"]{background-color:white; color:darkgrey}
                QPushButton:pressed {background-color:black; color:white}
                
                QLineEdit{background-color:lightblue;
                                /* border-style: outset; */
                                border-width: 2px;}
                
                /* QSplitter styling adopted from
                http://stackoverflow.com/questions/6832499/qsplitter-show-a-divider-or-a-margin-between-the-two-widgets
                */
                  
                QSplitter::handle:vertical {
                    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                                        stop:0 rgba(255, 255, 255, 0), 
                                        stop:0.407273 rgba(200, 200, 200, 255), 
                                        stop:0.4825 rgba(101, 104, 113, 235), 
                                        stop:0.6 rgba(255, 255, 255, 0));                 
                    height: 8px;
                    image: url(':/ellipses_v.svg');
                    }
                
                QSplitter::handle:horizontal {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                        stop:0 rgba(255, 255, 255, 0), 
                                        stop:0.407273 rgba(200, 200, 200, 255), 
                                        stop:0.4825 rgba(101, 104, 113, 235), 
                                        stop:0.6 rgba(255, 255, 255, 0)); 
                    width: 8px;
                    image: url(':/ellipses_h.svg');                     
                    }     
            """
# QApplication.setStyle(QStyleFactory.create('Cleanlooks')) re-create default styles

if THEME == 'dark':
    mpl_rc.update(mpl_rc_dark)
    params.update(mpl_params_dark)
    qss_rc = qss_common + qss_tab_bar + qss_dark
    
elif THEME == 'light':
    mpl_rc.update(mpl_rc_light)
    params.update(mpl_params_light)
    qss_rc = qss_common + qss_tab_bar + qss_light

elif THEME == 'none':
    mpl_rc.update(mpl_rc_light)
    params.update(mpl_params_light)  
    qss_rc = qss_common
    
else: # use the THEME name as the QStyle name 
    mpl_rc.update(mpl_rc_light)
    params.update(mpl_params_light)  
    qss_rc = THEME