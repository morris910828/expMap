#version 450 core

in vec2 TexCoord;

uniform sampler2D patchTex;
uniform int useTexture;        // 1 = 使用貼圖
uniform vec3 overrideColor;    // 0 = 顯示 highlight

out vec4 FragColor;

void main()
{
    if (useTexture == 1)
        FragColor = texture(patchTex, TexCoord);
    else
        FragColor = vec4(overrideColor, 1.0);
}
