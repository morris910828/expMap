#version 450 core

out vec4 FragColor;

uniform vec3 lineColor;        // 顏色
uniform float intensity;       // 亮度倍率（你可以控制）

void main()
{
    FragColor = vec4(lineColor * intensity, 1.0);
}
