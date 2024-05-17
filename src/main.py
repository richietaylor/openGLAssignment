import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import math


# Create and compile shaders
# def create_shader(vertex_filepath: str, fragment_filepath: str) -> int:


#     with open(vertex_filepath,'r') as f:
#         vertex_src = f.readlines()

#     with open(fragment_filepath,'r') as f:
#         fragment_src = f.readlines()
    
#     shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
#                             compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
#     return shader
class Shader:
    def __init__(self, vertex_filepath, fragment_filepath):
        self.program = self.create_shader(vertex_filepath, fragment_filepath)

    def create_shader(self, vertex_filepath, fragment_filepath):
        with open(vertex_filepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragment_filepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader

    def use(self):
        glUseProgram(self.program)

    def get_uniform_location(self, name):
        return glGetUniformLocation(self.program, name)

    def set_uniform_matrix4fv(self, name, matrix):
        glUniformMatrix4fv(self.get_uniform_location(name), 1, GL_FALSE, matrix)


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
    def __init__(self, position, eulers, scale=1, orbit_center=None, orbit_radius=0, orbit_speed=0, rotation_speed=[0, 0, 0]):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scale = scale
        self.orbit_center = np.array(orbit_center, dtype=np.float32) if orbit_center is not None else None
        self.rotation_speed = np.array(rotation_speed, dtype=np.float32)  
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.orbit_angle = 0

    # Updates the entity's position and rotation
    def update(self, delta_time):
        
        if self.orbit_center is not None and self.orbit_radius > 0:
            self.orbit_angle += self.orbit_speed * delta_time
            self.position[0] = self.orbit_center[0] + self.orbit_radius * math.cos(self.orbit_angle)
            self.position[1] = self.orbit_center[1] + self.orbit_radius * math.sin(self.orbit_angle)

            # self.eulers[0] += 1 * delta_time  
            # if self.eulers[1] > 360:
            #     self.eulers[1] -= 360
        self.eulers += self.rotation_speed * delta_time
        self.eulers %= 360 


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
        self.running = True
        # my_camera = Camera(entity=self.entities[0], distance=10, azimuth=0, elevation=0)
        
        # Orbiting light goes here because
        self.light_orbit_center = np.array([0.0, 0.0, -8.0], dtype=np.float32)
        self.light_orbit_radius = 10.0
        self.light_orbit_speed = 0.2
        self.light_orbit_angle = 0.0

        if self.entities:
            self.camera = Camera(entity=self.entities[0], distance=10, azimuth=0, elevation=0)
        else:
            # Fallback to a fixed position if no entities are available
            self.camera = Camera(entity=None, distance=10, azimuth=0, elevation=0)

    # def update_light_position(self, delta_time):
    #     self.light_orbit_angle += self.light_orbit_speed * delta_time
    #     self.lights[1]['position'][0] = self.light_orbit_center[0] + self.light_orbit_radius * math.cos(self.light_orbit_angle)
    #     self.lights[1]['position'][2] = self.light_orbit_center[2] + self.light_orbit_radius * math.sin(self.light_orbit_angle)
    def update_light_position(self, delta_time):
        self.light_orbit_angle += self.light_orbit_speed * delta_time
        self.lights[1]['position'][0] = self.light_orbit_center[0] + self.light_orbit_radius * math.cos(self.light_orbit_angle)
        self.lights[1]['position'][2] = self.light_orbit_center[2] + self.light_orbit_radius * math.sin(self.light_orbit_angle)

    # Initializes Pygame with OpenGL settings
    def _set_up_pygame(self) -> None:


        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((1920, 1080), pg.OPENGL|pg.DOUBLEBUF)


# Sets up the main application clock
    def _set_up_timer(self) -> None:
        self.clock = pg.time.Clock()

# Configures OpenGL rendering settings
    def _set_up_opengl(self) -> None:


        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)

    def _create_assets(self) -> None:

     # Define positions and files for multiple objects

        positions = [[0, 0, -8], [5, 0, -8], [10, 0, -8]] 
        eulers = [[0, 0, 0], [90, 0, 0], [0, 0, 0]]
        rotation_speeds = [[10,10,10],[0,0,60],[10,10,10]]
        scales = [0.5, 0.2, 0.1]
        models = ["sphere-fixed.obj", "sphere-fixed.obj", "sphere-fixed.obj"]
        textures = ["sun.png", "earth.png", "moon.png"]
        orbit_params = [
            (None, 0, 0),           
            ([0, 0, -8], 2, 0.5),     
            (None, 0.5, 3)      
        ]


        for pos, euler, rotation_speed,scale, model_file, texture_file, (orbit_center, radius, speed) in zip(positions, eulers, rotation_speeds, scales, models, textures, orbit_params):
            entity = Entity(position=pos, eulers=euler, rotation_speed=rotation_speed, scale=scale, orbit_center=orbit_center, orbit_radius=radius, orbit_speed=speed, )
            mesh = Mesh(f"resources/{model_file}")
            material = Material(f"resources/{texture_file}")
            self.entities.append(entity)
            self.meshes.append(mesh)
            self.materials.append(material)

        # self.shader = create_shader(
        #     vertex_filepath = "shaders/simple.vert", 
        #     fragment_filepath = "shaders/simple.frag")
        self.shader = Shader(vertex_filepath="shaders/simple.vert", fragment_filepath="shaders/simple.frag")
        
        

        # self.lights = [
        #     {'position': [0.0, 0.0, -8.0], 'color': [1.0, 1.0, 0.8]},  # Sun
        #     {'position': [10.0, 10.0, 10.0], 'color': [0.8, 0.8, 1.0]}  # Additional light
        # ]
        
        # # Material properties
        # self.material_ambient = [0.1, 0.1, 0.1]
        # self.material_diffuse = [0.6, 0.6, 0.6]
        # self.material_specular = [0.5, 0.5, 0.5]
        # self.material_shininess = 32.0        

        self.lights = [
        {'position': [0.0, 0.0, -8.0], 'color': [5.0,5.0,1.0]},  # Sun
        {'position': [10.0, 0.0, -8.0], 'color': [5.0,5.0,5.0]}   # Additional light
        ]

        self.material_ambient = [0.3, 0.3, 0.3]
        self.material_diffuse = [0.7, 0.7, 0.7]
        self.material_specular = [1.0, 1.0, 1.0]
        self.material_shininess = 32.0

         
    # Sets projection matrix and other one-time OpenGL settings
    def _set_onetime_uniforms(self) -> None:

        # glUseProgram(self.shader)
        self.shader.use()
        glUniform1i(glGetUniformLocation(self.shader.program, "imageTexture"), 0)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 1920/1080, 
            near = 0.1, far = 50, dtype=np.float32
        )
        # glUniformMatrix4fv(
        #     glGetUniformLocation(self.shader,"projection"),
        #     1, GL_FALSE, projection_transform
        # )
        self.shader.set_uniform_matrix4fv("projection", projection_transform)


        for i, light in enumerate(self.lights):
            glUniform3fv(glGetUniformLocation(self.shader.program, f"lights[{i}].position"), 1, light['position'])
            glUniform3fv(glGetUniformLocation(self.shader.program, f"lights[{i}].color"), 1, light['color'])

        # Pass material properties
        glUniform3fv(glGetUniformLocation(self.shader.program, "materialAmbient"), 1, self.material_ambient)
        glUniform3fv(glGetUniformLocation(self.shader.program, "materialDiffuse"), 1, self.material_diffuse)
        glUniform3fv(glGetUniformLocation(self.shader.program, "materialSpecular"), 1, self.material_specular)
        glUniform1f(glGetUniformLocation(self.shader.program, "materialShininess"), self.material_shininess)

    
    # Retrieves and stores uniform locations from the shader
    def _get_uniform_locations(self) -> None:
        self.shader.use()
        self.modelMatrixLocation = glGetUniformLocation(self.shader.program, "model")


        # glUseProgram(self.shader)
        # self.modelMatrixLocation = glGetUniformLocation(self.shader,"model")
    
    # Main loop that updates and renders the game
    def run(self):
            keep_running = True
            anim_running = True
            target_entity = 0
            last_time = pg.time.get_ticks()
            while keep_running:
                current_time = pg.time.get_ticks()
                delta_time = (current_time - last_time) / 1000.0  # seconds
                last_time = current_time

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        keep_running = False
                    elif event.type == pg.KEYDOWN:
                        if anim_running:
                            earth_orbit = self.entities[1].orbit_speed
                            moon_orbit = self.entities[2].orbit_speed
                            sun_rotation = self.entities[0].rotation_speed
                            earth_rotation = self.entities[1].rotation_speed                            
                            moon_rotaion = self.entities[2].rotation_speed

                        if event.key == pg.K_SPACE:  # Toggle animation on/off
                            if anim_running:
                                self.entities[1].orbit_speed = 0
                                self.entities[2].orbit_speed = 0
                                self.entities[0].rotation_speed = 0
                                self.entities[1].rotation_speed = 0
                                self.entities[2].rotation_speed = 0
                            else:
                                self.entities[1].orbit_speed = earth_orbit
                                self.entities[2].orbit_speed = moon_orbit
                                self.entities[0].rotation_speed = sun_rotation
                                self.entities[1].rotation_speed = earth_rotation
                                self.entities[2].rotation_speed = moon_rotaion

                            anim_running = not anim_running
                            
                        if event.key == pg.K_TAB:
                            target_entity+=1
                            if target_entity> len(self.entities)-1:
                                target_entity=0
                            self.camera.set_target(self.entities[target_entity])
                            # print(f"Camera now targeting entity {current_target_index}")
                        # if event.key == pg.K_LEFT:
                        #     self.camera.update(d_azimuth=-1)  # Rotate left around the target
                        # elif event.key == pg.K_RIGHT:
                        #     self.camera.update(d_azimuth=1)   # Rotate right around the target
                        # elif event.key == pg.K_UP:
                        #     self.camera.update(d_elevation=1)  # Rotate up around the target
                        # elif event.key == pg.K_DOWN:
                        #     self.camera.update(d_elevation=-1) # Rotate down around the target
                        # elif event.key == pg.K_PAGEUP:
                        #     self.camera.update(d_distance=-0.5)  # Zoom in
                        # elif event.key == pg.K_PAGEDOWN:
                        #     self.camera.update(d_distance=0.5)   # Zoom out
                
                # Check for continuous key presses
                keys = pg.key.get_pressed()
                if keys[pg.K_a]:
                    self.camera.update(d_azimuth=-2)  # Rotate left around the target
                if keys[pg.K_d]:
                    self.camera.update(d_azimuth=2)   # Rotate right around the target
                if keys[pg.K_w]:
                    self.camera.update(d_elevation=2)  # Rotate up around the target
                if keys[pg.K_s]:
                    self.camera.update(d_elevation=-2) # Rotate down around the target
                if keys[pg.K_LEFT]:
                    self.camera.update(d_distance=-0.5)  # Zoom in
                if keys[pg.K_RIGHT]:
                    self.camera.update(d_distance=0.5)   # Zoom out

                if keys[pg.K_j]:
                    self.entities[1].orbit_speed+=0.1
                if keys[pg.K_n]:
                    self.entities[1].orbit_speed+=-0.1

                if keys[pg.K_k]:
                    self.entities[2].orbit_speed+=0.1
                if keys[pg.K_m]:
                    self.entities[2].orbit_speed+=-0.1


                if self.camera:  
                        view_matrix = self.camera.get_view_matrix()
                        glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "view"), 1, GL_FALSE, view_matrix)
                        glUniform3fv(glGetUniformLocation(self.shader.program, "viewPos"), 1, self.camera.position)
                
                if self.running:
                    for entity in self.entities:
                        entity.update(delta_time)

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glUseProgram(self.shader.program)

                    for entity, mesh, material in zip(self.entities, self.meshes, self.materials):
                        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, entity.get_model_transform())
                        material.use()
                        mesh.arm_for_drawing()
                        mesh.draw()

                    for i, light in enumerate(self.lights):
                        glUniform3fv(glGetUniformLocation(self.shader.program, f"lights[{i}].position"), 1, light['position'])
                        glUniform3fv(glGetUniformLocation(self.shader.program, f"lights[{i}].color"), 1, light['color'])


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
        glDeleteProgram(self.shader.program)
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


class Camera:
    def __init__(self, entity, distance, azimuth, elevation):
        self.entity = entity  # The target is now an entity
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.calculate_position()

    def set_target(self, new_entity):
        self.entity = new_entity
        self.calculate_position()

    def calculate_position(self):
        if self.entity:
            self.target = self.entity.position 
        else:
            self.target = np.array([0, 0, 0], dtype=np.float32)  

        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)

        x = self.distance * np.cos(el) * np.sin(az)
        y = self.distance * np.sin(el)
        z = self.distance * np.cos(el) * np.cos(az)

        self.position = self.target + np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(self.position, self.target, self.up, dtype=np.float32)

    def update(self, d_azimuth=0, d_elevation=0, d_distance=0):
        self.azimuth += d_azimuth
        self.elevation = max(-89, min(89, self.elevation + d_elevation))
        self.distance = max(1, min(50, self.distance + d_distance))
        self.calculate_position()


my_app = App()
my_app.run()
my_app.quit()