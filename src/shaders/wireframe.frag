#version 450 core

uniform vec3 lineColor;   // main.cpp 會傳入 (1,0,0)

out vec4 FragColor;

void main()
{
    FragColor = vec4(lineColor, 1.0);
}
