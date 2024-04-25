#version 330 core

uniform vec3 objectColor;

out vec4 outColor;

void main()
{
    outColor = vec4(objectColor,1);
}
