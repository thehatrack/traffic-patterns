"""
VideoStreams Module

This module contains helpers that simplify the handling of streaming
video from remote sources, such as TCP/IP cameras.
"""
import time
import threading
from datetime import datetime

import cv2
from PIL import Image


class VideoSource(object):
    """Manage the connection to a remote video source."""
    
    def __init__(self, url):
        """Initialize the object."""
        self._url = url
        self._thread = None
        self._stream = None
        self._connecting = False
        self._last_frame = None
        self._last_frame_time = None
        self._next = None
        
    def __del__(self):
        """Stop the stream when the object is removed from memory."""
        self.stop_stream()
        
    def snap(self):
        """Return the most recent frame from the video source with timestamp."""
        # returns (timestamp, image)
        if self._last_frame is not None:
            return (
                self._last_frame_time,
                cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
            )
        else:
            return (None, None)
    
    def reconnect(self):
        """Reconnect to the video source after an interruption."""
        while self._connecting:
            time.sleep(1)
                
        self._connecting = True
        
        connected = False
        while not connected:
            self._stream = cv2.VideoCapture(self._url)
            connected = self._stream.isOpened()
            if connected:
                self._connecting = False
                return
            
            print('Could not connect to {}'.format(self._url))
            print('Retrying in 1 minute')
            
            time.sleep(60)
            
    def start_stream(self):
        """Connect to the video source and start pulling in frames."""
        if self._thread and self._thread.is_alive():
            print('Thread already running')
            return
        
        self.reconnect()
        self._thread = threading.Thread(target=self._watch, args=())
        self._thread.start()
        
    def stop_stream(self):
        """Disconnect from the video source."""
        self._stream.release()
        
    def _watch(self):
        """While the video stream is alive, cache the most recent frame."""
        fail_counter = 0
        while self._stream.isOpened():
            frame_time = datetime.now()
            ret, frame = self._stream.read()
            
            if not ret:
                fail_counter += 1
            else:
                fail_counter = 0
                self._last_frame_time = frame_time
                self._last_frame = frame
            
            time.sleep(1/60)
                
            if fail_counter > 50:
                self.reconnect()
                fail_counter = 0        
                
        self._stream.release()
        
        
def stream_factory(url):
    """Create an interator for the handling of camera frames."""
    camera = VideoSource(url)
    camera.start_stream()
    while True:
        yield camera.snap()