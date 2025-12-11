#version 450 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D patchTex;
uniform int useTexture;
uniform vec3 overrideColor;

void main()
{
    if (useTexture == 1)
        FragColor = texture(patchTex, TexCoord);
    else
        FragColor = vec4(overrideColor, 1.0);
}
