# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:12:26 2018

@author: JakobLab
"""
import sys
import time
import numpy as np
import collections

class Progbar():
    """Displays a progress bar.
    
    Arguments
    ---------
    
    target : int 
        Total number of steps expected, None if unknown.
    width : int , optional (default=30)
        Progress bar width on screen. 
    verbose : int , optional (default=1)
        Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose). 
    interval : float , optional (default=0.05)
        Minimum visual progress update interval (in seconds).
    caps : list of str , optional (default=['[',']'])
        Start and End caps of progressbar
    passed_marker : str , optional (default='=')
        Symbol of passed section
    missing_marker : str , optional (default='0')
        Symbol of missing section
    current_marker : str , optional (default='>')
        Symbol of current section
    newline_on_end : boolean , optional (default=True)
        Adds a newline after each progressbar
    text_description : str ,optional (default='')
        Adds a textstring in front of the progressbar
    
    Attributes
    ----------
    
    target : int
        Total number of steps expected, None if unknown.
    width : int
        Progress bar width on screen. 
    verbose : int
        Verbosity mode
    interval : float
        Minimum visual progress update interval (seconds)
    newline_on_end : boolean
        True if '\n' is added when finished
    start_cap : str
        Start cap for the progress bar
    end_cap : str
        End cap for the progress bar
    passed : str
        Symbol for passed progress
    missing : str
        Symbol for missing progress
    current : str
        Symbol for current progress
    text : str
        String in front of progress bar
    last_run :float
        Time for last update
        
    """

    def __init__(self, target, width=30, verbose=1,
                 interval=0.05, caps=['▐','▌'], passed_marker='█',
                 missing_marker='░', current_marker='▓',
                 newline_on_end=True, text_description=''):
        
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.newline_on_end = newline_on_end
        self.start_cap = caps[0]
        self.end_cap = caps[1]
        self.passed = passed_marker
        self.missing = missing_marker
        self.text = text_description
        self.current = current_marker
        self.last_run = 0
        
        self._dynamic_display = ((hasattr(sys.stderr, 'isatty') and
                                  sys.stderr.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0
        

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        self._seen_so_far = current
        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stderr.write('\b' * prev_total_width)
                sys.stderr.write('\r')
            else:
                sys.stderr.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = self.text
                barstr +=' %%%dd/%d ' % (numdigits, self.target)
                barstr += self.start_cap
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += (self.passed * (prog_width - 1))
                    if current < self.target:
                        bar += self.current
                    else:
                        bar += self.passed
                bar += (self.missing * (self.width - prog_width))
                bar += self.end_cap
            else:
                bar = self.text+' %7d/Unknown' % current

            self._total_width = len(bar)
            sys.stderr.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                if self.newline_on_end:
                    info += '\n'
                self.last_run = now-self._start

            sys.stderr.write(info)
            sys.stderr.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)