#version 120
varying vec2 fragTex;
uniform sampler2D alphaTexture;
uniform vec4 color;

void main()
{
	float alpha = texture2D(alphaTexture, fragTex).r;
	gl_FragColor = vec4(color.rgb, color.a*alpha);
}
