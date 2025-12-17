#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

void main()
{
    vec4 texColor = texture(texture1, TexCoord);
    
    // [修改] 如果貼圖有透明部分 (Alpha < 0.1)，則丟棄該像素 (去背效果)
    if(texColor.a < 0.1)
        discard;
        
    // 直接輸出貼圖顏色，不加任何濾鏡
    FragColor = texColor;
}