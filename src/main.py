import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import math

# Create and compile shaders
def create_shader(vertex_filepath: str, fragment_filepath: str) -> int:


    with open(vertex_filepath,'r') as f:
        vertex_src = f.readlines()

    with open(fragment_filepath,'r') as f:
        fragment_src = f.readlines()
    
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
    return shader


# load a mesh from an OBJ file
def loadMesh(filename: str) -> list[float]:

    v = []
    vt = []
    vn = []

    vertices = []

    with open(filename, "r") as file:

        line = file.readline()

        while line:

            words = line.split()
            if words[0] == "v":
                v.append(read_vertex_data(words))
            elif words[0] == "vt":
                vt.append(read_texcoord_data(words))
            elif words[0] == "vn":
                vn.append(read_normal_data(words))
            elif words[0] == "f":
                read_face_data(words, v, vt, vn, vertices)    
            line = file.readline()

    return vertices

 # Helper functions to parse vertex, texture, and normal data from OBJ file  

def read_vertex_data(words: list[str]) -> list[float]:


    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]
    
def read_texcoord_data(words: list[str]) -> list[float]:

    return [
        float(words[1]),
        float(words[2])
    ]
    
def read_normal_data(words: list[str]) -> list[float]:

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]

# Function to process face data and link vertices, textures, and normals

def read_face_data(
    words: list[str], 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:

    triangleCount = len(words) - 3

    for i in range(triangleCount):

        make_corner(words[1], v, vt, vn, vertices)
        make_corner(words[2 + i], v, vt, vn, vertices)
        make_corner(words[3 + i], v, vt, vn, vertices)
    
# Combines vertex, texture, and normal data into a single array for OpenGL

def make_corner(corner_description: str, 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:


    v_vt_vn = corner_description.split("/")
    
    for element in v[int(v_vt_vn[0]) - 1]:
        vertices.append(element)
    for element in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(element)
    for element in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(element)

# Represents an object in 3D space, including its position, rotation, and scale
class Entity:
    def __init__(self, position, eulers, scale=1, orbit_center=None, orbit_radius=0, orbit_speed=0):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scale = scale
        self.orbit_center = np.array(orbit_center, dtype=np.float32) if orbit_center is not None else None
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.orbit_angle = 0

    # Updates the entity's position and rotation
    def update(self, delta_time):
        
        if self.orbit_center is not None and self.orbit_radius > 0:
            self.orbit_angle += self.orbit_speed * delta_time
            self.position[0] = self.orbit_center[0] + self.orbit_radius * math.cos(self.orbit_angle)
            self.position[1] = self.orbit_center[1] + self.orbit_radius * math.sin(self.orbit_angle)


        
        self.eulers[1] += 0.25 * delta_time  
        if self.eulers[1] > 360:
            self.eulers[1] -= 360


    # Computes the transformation matrix for rendering
    def get_model_transform(self):
        # Scale -> Rotate -> Translate
        scale_matrix = pyrr.matrix44.create_from_scale([self.scale, self.scale, self.scale], dtype=np.float32)
        rotation_matrix = pyrr.matrix44.create_from_eulers(np.radians(self.eulers), dtype=np.float32)
        translation_matrix = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)

        # Start with translation, then rotate, and finally scale (note the reverse application order)
        model_transform = pyrr.matrix44.multiply(scale_matrix, rotation_matrix)
        # Then apply translation
        model_transform = pyrr.matrix44.multiply(model_transform, translation_matrix)

        return model_transform


# Main application class managing game state
class App:


    def __init__(self):

        self._set_up_pygame()
        self._set_up_timer()
        self._set_up_opengl()
        self.entities = []
        self.meshes = []
        self.materials = []
        self._create_assets()
        self._set_onetime_uniforms()
        self._get_uniform_locations()
        self.running = False
    
    # Initializes Pygame with OpenGL settings
    def _set_up_pygame(self) -> None:


        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((1280 ,720), pg.OPENGL|pg.DOUBLEBUF)


# Sets up the main application clock
    def _set_up_timer(self) -> None:
        self.clock = pg.time.Clock()

# Configures OpenGL rendering settings
    def _set_up_opengl(self) -> None:


        glClearColor(0.1, 0.2, 0.2, 1)
        glEnable(GL_DEPTH_TEST)

    def _create_assets(self) -> None:

     # Define positions and files for multiple objects

        positions = [[0, 0, -8], [5, 0, -8], [10, 0, -8]] 
        eulers = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        scales = [0.5, 0.2, 0.1]
        models = ["sphere-fixed.obj", "sphere-fixed.obj", "sphere-fixed.obj"]
        textures = ["yellow.png", "blue.png", "grey.jpeg"]
        orbit_params = [
            (None, 0, 0),           
            ([0, 0, -8], 2, 0.5),     
            (None, 0.5, 3)      
        ]


        for pos, euler, scale, model_file, texture_file, (orbit_center, radius, speed) in zip(positions, eulers, scales, models, textures, orbit_params):
            entity = Entity(position=pos, eulers=euler, scale=scale, orbit_center=orbit_center, orbit_radius=radius, orbit_speed=speed)
            mesh = Mesh(f"resources/{model_file}")
            material = Material(f"resources/{texture_file}")
            self.entities.append(entity)
            self.meshes.append(mesh)
            self.materials.append(material)

        self.shader = create_shader(
            vertex_filepath = "shaders/simple.vert", 
            fragment_filepath = "shaders/simple.frag")
         
    # Sets projection matrix and other one-time OpenGL settings
    def _set_onetime_uniforms(self) -> None:

        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 1280/720, 
            near = 0.1, far = 50, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )
    
    # Retrieves and stores uniform locations from the shader
    def _get_uniform_locations(self) -> None:


        glUseProgram(self.shader)
        self.modelMatrixLocation = glGetUniformLocation(self.shader,"model")
    
    # Main loop that updates and renders the game
    def run(self):
            keep_running = True
            last_time = pg.time.get_ticks()
            while keep_running:
                current_time = pg.time.get_ticks()
                delta_time = (current_time - last_time) / 1000.0  # seconds
                last_time = current_time

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        keep_running = False
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_SPACE:  # Toggle animation on/off
                            self.running = not self.running
                        elif event.key == pg.K_UP:
                            self.entities[1].orbit_speed += 0.1  # Increase orbit speed of second object
                        elif event.key == pg.K_DOWN:
                            self.entities[1].orbit_speed -= 0.1  # Decrease orbit speed of second object
                        elif event.key == pg.K_RIGHT:
                            self.entities[2].orbit_speed += 0.1  # Increase orbit speed of third object
                        elif event.key == pg.K_LEFT:
                            self.entities[2].orbit_speed -= 0.1  # Decrease orbit speed of third object
                        elif event.key == pg.K_q:
                            keep_running = False

                if self.running:
                    for entity in self.entities:
                        entity.update(delta_time)

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glUseProgram(self.shader)

                    for entity, mesh, material in zip(self.entities, self.meshes, self.materials):
                        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, entity.get_model_transform())
                        material.use()
                        mesh.arm_for_drawing()
                        mesh.draw()

                    if len(self.entities) > 2:
                        self.entities[2].orbit_center = np.copy(self.entities[1].position)

                pg.display.flip()
                self.clock.tick(60)  # This controls frame rate; it might be wise to separate drawing and updating rates.

# Cleanup function to free resources
    def quit(self) -> None:


        for mesh in self.meshes:
            mesh.destroy()
        for material in self.materials:
            material.destroy()
        glDeleteProgram(self.shader)
        pg.quit()

class Mesh:

    def __init__(self, filename: str):

        vertices = loadMesh(filename)
        self.vertex_count = len(vertices)//8
        vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        # Position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        # Texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
    
    def arm_for_drawing(self) -> None:
        glBindVertexArray(self.vao)
    
    def draw(self) -> None:

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self) -> None:
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))
    
    def destroy(self) -> None:        
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class Material:

    
    def __init__(self, filepath: str):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self) -> None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self) -> None:
        glDeleteTextures(1, (self.texture,))

my_app = App()
my_app.run()
my_app.quit()