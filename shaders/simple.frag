#version 330 core

in vec2 fragTexCoords;
in vec3 fragNormal;
in vec3 fragPos;

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
    vec3 ambient = vec3(0.0);
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    vec3 norm = normalize(fragNormal);
    vec3 viewDir = normalize(viewPos - fragPos);

    for (int i = 0; i < 2; i++) {
        vec3 lightDir = normalize(lights[i].position - fragPos);
        
        
        ambient += materialAmbient * lights[i].color * 0.3; 

        
        float diff = max(dot(norm, lightDir), 0.0);
        diffuse += materialDiffuse * diff * lights[i].color * 1.0; 

        
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
        specular += materialSpecular * spec * lights[i].color * 0.5; 
    }

    vec3 lighting = ambient + diffuse + specular;
    vec4 texColor = texture(imageTexture, fragTexCoords);
    color = texColor * vec4(lighting, 1.0);
}
