#version 450 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 viewPos;

void main()
{
    // -----------------------
    // 1. Depth shading (near/far)
    // -----------------------
    float depth = gl_FragCoord.z;
    
    // 調亮版本：depth 影響程度降低
    float depthShade = 1.0 - depth * 0.4;   // 原本 0.6 改成 0.25

    // -----------------------
    // 2. Edge shading (輪廓)
    // -----------------------
    vec3 V = normalize(viewPos - FragPos);
    float edge = abs(dot(normalize(Normal), V));

    // 讓邊緣暗一點、但別太過頭
    float edgeShade = pow(edge, 0.8);   // 原本 0.3 → 0.55，比較亮

    // -----------------------
    // 3. Combine
    // -----------------------
    float shade = depthShade * edgeShade;

    // 整體再拉亮一些
    shade = shade * 0.9 + 0.4;       // 提亮

    vec3 color = vec3(shade);
    FragColor = vec4(color, 1.0);
}
