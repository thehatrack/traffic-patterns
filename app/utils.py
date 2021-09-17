import datetime
import sys

import jinja2


class BlockTimer(object):
    enter_template = jinja2.Template('Starting {{ name }}... ')
    exit_template = jinja2.Template('done [{{ (end - start).total_seconds() }} seconds]\n\n')
    
    def __init__(self, name):
        self._start = None
        self._name = name
        
    def __enter__(self):
        self._start = datetime.datetime.now()
        sys.stdout.write(
            self.enter_template.render(name=self._name,
                                       start=self._start,
                                       end=self._start))
        
    def __exit__(self, type, value, traceback):
        sys.stdout.write(
            self.exit_template.render(name=self._name,
                                      start=self._start,
                                      end=datetime.datetime.now()))
        