#version 330 core

in vec2 fragTexCoords;
in vec3 fragNormal;
in vec3 fragPosition;

out vec4 color;

struct Light {
    vec3 position;
    vec3 color;
};

uniform sampler2D imageTexture;
uniform Light lights[2];
uniform vec3 viewPos;

uniform vec3 materialAmbient;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform float materialShininess;

void main()
{
    // Ambient lighting
    vec3 ambient = materialAmbient * lights[0].color;

    // Diffuse lighting
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lights[0].position - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = materialDiffuse * diff * lights[0].color;

    // Specular lighting
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
    vec3 specular = materialSpecular * spec * lights[0].color;

    vec3 result = ambient + diffuse + specular;

    color = texture(imageTexture, fragTexCoords) * vec4(result, 1.0);
}
