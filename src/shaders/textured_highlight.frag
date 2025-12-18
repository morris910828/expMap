#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D texture1;
uniform vec3 viewPos;

void main()
{
    bool isInside = (TexCoord.x >= 0.0 && TexCoord.x <= 1.0 &&
                     TexCoord.y >= 0.0 && TexCoord.y <= 1.0);
    if (!isInside) discard;

    vec4 texColor = texture(texture1, TexCoord);
    if(texColor.a < 0.1) discard;

    // ==========================================
    // 光照計算
    // ==========================================
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8)); 

    float ambientStrength = 0.8; 
    vec3 ambient = ambientStrength * vec3(1.0);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(0.4); 

    vec3 lighting = (ambient + diffuse);

    // lighting = min(lighting, vec3(1.2)); 

    vec3 finalColor = texColor.rgb * lighting;

    FragColor = vec4(finalColor, texColor.a);
}