import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# Just keep this out for the moment - helps focus the mind
from Geometry import Geometry


class Triangle:

    def __init__(self, shader):
        self.vertexLoc = glGetAttribLocation(shader, "position")
        self.vertices = np.array([0.0, 0.5, 0.0,
                                  -0.5, -0.5, 0.0,
                                  0.5, -0.5, 0.0,

                                  0.5, -0.5, 0.0,
                                  -0.5, -0.5, 0.0,
                                  -0.5, 0.5, 0.0], dtype=np.float32)

        self.vertexCount = 6
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(self.vertexLoc)
        glVertexAttribPointer(self.vertexLoc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    def cleanup(self):
        glDeleteBuffers(1, (self.vbo,))


class OpenGLWindow:

    def __init__(self):
        self.triangle = None
        self.clock = pg.time.Clock()

    def loadShaderProgram(self, vertex, fragment):
        with open(vertex, 'r') as f:
            vertex_src = f.readlines()

        with open(fragment, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader

    def initGL(self, screen_width=2560 , screen_height=1440):
        pg.init()

        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 2)

        pg.display.set_mode((screen_width, screen_height), pg.OPENGL | pg.DOUBLEBUF)

        glEnable(GL_DEPTH_TEST)
        # Uncomment these two lines when perspective camera has been implemented
        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)
        glClearColor(0.2, 0, 0, 1)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Note that this path is relative to your working directory when running the program
        # You will need change the filepath if you are running the script from inside ./src/

        self.shader = self.loadShaderProgram("./shaders/simple.vert", "./shaders/simple.frag")
        glUseProgram(self.shader)

        colorLoc = glGetUniformLocation(self.shader, "objectColor")
        glUniform3f(colorLoc, 0.4, 1.0, 1.0)

        # Uncomment this for triangle rendering
        self.triangle = Triangle(self.shader)

        # Uncomment this for model rendering
        # self.cube = Geometry('./resources/prism.obj')

        print("Setup complete!")


    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)  # You may not need this line

        #Uncomment this for triangle rendering
        glDrawArrays(GL_TRIANGLES, 0, self.triangle.vertexCount)

        # Uncomment this for model rendering
        # glDrawArrays(GL_TRIANGLES, 0, self.cube.vertexCount)


        # Swap the front and back buffers on the window, effectively putting what we just "drew"
        # Onto the screen (whereas previously it only existed in memory)
        pg.display.flip()

    def cleanup(self):
        glDeleteVertexArrays(1, (self.vao,))
        # Uncomment for triangle rendering
        self.triangle.cleanup()
        # Uncomment for model rendering
        # self.cube.cleanup()
