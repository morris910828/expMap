#version 330 core
out vec4 FragColor;

in vec2 TexCoords; // 接收 UV

uniform vec3 color;
uniform sampler2D decalTexture;
uniform int useTexture;

void main()
{
    if (useTexture == 1) {
        vec4 texColor = texture(decalTexture, TexCoords);
        // 如果貼圖有透明通道，可以 discard
        if(texColor.a < 0.1) discard;
        FragColor = texColor; 
    } else {
        FragColor = vec4(color, 0.7);
    }
}