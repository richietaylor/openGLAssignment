#version 330 core

layout (location = 0) in vec3 aPos;       // Vertex position
layout (location = 1) in vec2 aTexCoords; // Texture coordinates
layout (location = 2) in vec3 aNormal;    // Vertex normal

out vec3 fragPos;   // Position of the fragment
out vec3 fragNormal;    // Normal of the fragment
out vec2 fragTexCoords; // Texture coordinates

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    fragPos = vec3(model * vec4(aPos, 1.0));
    fragNormal = normalize(mat3(transpose(inverse(model))) * aNormal); // Correct normal transformation
    fragTexCoords = aTexCoords;
    
    gl_Position = projection * view * vec4(fragPos, 1.0);
}
