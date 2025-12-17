#version 450 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 viewPos;

void main()
{
    float depth = gl_FragCoord.z;
    
    float depthShade = 1.0 - depth * 0.4;

    vec3 V = normalize(viewPos - FragPos);
    float edge = abs(dot(normalize(Normal), V));
    float edgeShade = pow(edge, 0.8);

    float shade = depthShade * edgeShade;

    shade = shade * 0.9 + 0.4;

    vec3 color = vec3(shade);
    FragColor = vec4(color, 1.0);
}