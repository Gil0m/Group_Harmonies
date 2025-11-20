# render_pyglet.py
import pyglet
from pyglet.gl import *
import numpy as np
from boids import Flock


class Renderer:
    def __init__(self, flock: Flock, window_size=(800,600)):
        self.flock = flock
        self.win = pyglet.window.Window(width=window_size[0], height=window_size[1], resizable=True)
        glEnable(GL_DEPTH_TEST)
        # camera params
        self.camera_distance = max(flock.bounds) * 3.0


    @self.win.event
    def on_draw(self):
        self.win.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        w,h = self.win.get_size()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w/float(h), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # simple camera lookat
        gluLookAt(self.camera_distance, self.camera_distance, self.camera_distance,
        0,0,0, 0,0,1)
        self.draw_boids()


    def draw_boids(self):
        positions, velocities = self.flock.get_state_arrays()
        glPointSize(6.0)
        glBegin(GL_POINTS)
        for p in positions:
            glVertex3f(p[0], p[1], p[2])
            glEnd()


    def run(self, fps=60):
        def update(dt):
            self.flock.step()
        pyglet.clock.schedule_interval(update, 1.0/fps)
        pyglet.app.run()


if __name__ == '__main__':
    flock = Flock(n=80, bounds=[20,20,20], seed=42)
    r = Renderer(flock)
    r.run() 