#version 450 core

in vec2 TexCoord;

uniform sampler2D patchTex;
uniform int useTexture;
uniform vec3 overrideColor;

out vec4 FragColor;

void main()
{
    if (useTexture == 1)
    {
        FragColor = texture(patchTex, TexCoord);
    }
    else
    {
        FragColor = vec4(overrideColor, 1.0);
    }
}
