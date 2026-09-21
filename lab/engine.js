/* ==========================================================================
   DEPTH//LAB ENGINE
   One WebGL2 pipeline that runs every effect, driven by a flat state object `S`.
   Pass 1  scene      image on a displaced mesh / point field  -> colour + aux(depth,uv)
   Pass 2  focus      depth-of-field blur                       -> colour
   Pass 3  composite  sort, fog, contours, scan, lens, dither, mosh -> screen
   ========================================================================== */

const clamp=(x,a,b)=>Math.min(b,Math.max(a,x)),lerp=(a,b,t)=>a+(b-a)*t;
const FOV=40*Math.PI/180,TAN=Math.tan(FOV/2),P11=1/TAN,HIST=64,BG=[.039,.039,.039];

export const DEFAULTS={
  // geometry + camera
  geometry:'mesh',relief:.6,fill:1.05,round:0,zoom:1,yaw:0,pitch:0,cx:0,cy:0,sway:.35,swaySpeed:.5,
  // 01 rack focus
  focusOn:0,focus:.6,aperture:3,focusFollow:1,
  // 02 scan reveal
  scanOn:0,scan:.5,scanAuto:.25,scanBand:.03,
  // 03 relight
  lightOn:0,light:1,lightH:.35,ambient:.22,spec:.35,lightWarm:.5,lightFollow:1,lightX:.5,lightY:.35,
  // 04 fog
  fogOn:0,fog:.65,fogNear:.15,fogFar:.85,fogSpeed:.3,fogTone:.82,
  // 05 dissolve + morph
  dissolveOn:0,dissolve:.5,turb:.35,wind:.25,morph:0,
  // 06 depth type
  textOn:0,text:'FIELD',textSize:.34,textY:.42,textDepth:.45,textColor:0,
  // 07 slit-scan
  slitOn:0,slit:.6,slitAxis:0,
  // 08 pixel sort + datamosh
  sortOn:0,sortThr:.5,sortLen:.5,sortDir:0,moshOn:0,mosh:.5,moshBlock:24,
  // 09 lens
  lensOn:0,lensMode:0,lensR:.2,lensFollow:1,lensX:.5,lensY:.5,
  // 10 contours
  contourOn:0,contour:.8,contourN:26,contourSpeed:.08,contourDim:.55,
  // finish
  ditherOn:0,dither:.85,ditherLv:4,mono:0,scanlines:0,
  // 11 gyro, 12 video handoff
  gyroOn:0,videoOn:0,videoMix:0,
};

/* ---------------------------------------------------------------- image helpers */
export function mk(w,h){const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;return{canvas,ctx:canvas.getContext('2d',{willReadFrequently:false})}}
export const loadImg=src=>new Promise((ok,no)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=()=>no(new Error('load failed: '+src));i.src=src});
export function toCanvas(img,max=2048){const W=img.naturalWidth||img.videoWidth||img.width,H=img.naturalHeight||img.videoHeight||img.height,s=Math.min(1,max/Math.max(W,H)),m=mk(Math.round(W*s),Math.round(H*s));m.ctx.drawImage(img,0,0,m.canvas.width,m.canvas.height);return m.canvas}
export function lumaDepth(src){const m=mk(Math.round(src.width/2),Math.round(src.height/2));m.ctx.filter='grayscale(1) blur(3px)';m.ctx.drawImage(src,0,0,m.canvas.width,m.canvas.height);return m.canvas}
/* Depth levelling. Estimated depth is inverse distance, so far scenes crowd into the bottom of the range
   (a mountain plate may only use 0-0.16). Stretch to the used range and blend toward an equalised curve so
   relief, fog, focus and type depth all get usable separation. Returns the stretch factor so smoothing can match. */
export function levelDepth(D,amt=.55){
  const w=D.width,h=D.height,m=mk(w,h);m.ctx.drawImage(D,0,0);
  if(amt<=0)return{canvas:m.canvas,stretch:1};
  const im=m.ctx.getImageData(0,0,w,h),p=im.data,hist=new Uint32Array(256),cdf=new Float32Array(256);
  for(let i=0;i<p.length;i+=4)hist[p[i]]++;
  for(let i=0,acc=0;i<256;i++){acc+=hist[i];cdf[i]=acc/(w*h)}
  let lo=0,hi=255;while(lo<253&&cdf[lo]<.004)lo++;while(hi>lo+2&&cdf[hi-1]>.996)hi--;
  const lut=new Uint8ClampedArray(256);for(let i=0;i<256;i++)lut[i]=255*lerp(clamp((i-lo)/(hi-lo),0,1),cdf[i],amt);
  for(let i=0;i<p.length;i+=4)p[i]=p[i+1]=p[i+2]=lut[p[i]];
  m.ctx.putImageData(im,0,0);return{canvas:m.canvas,stretch:255/(hi-lo)};
}
/* grow near shapes a little, then blur: mesh stretch lands on background, not on the subject outline */
export function soften(D,k=1,stretch=1){
  const m=mk(D.width,D.height),c=m.ctx,r=Math.max(1.5,D.width/380)*k;
  c.drawImage(D,0,0);c.globalCompositeOperation='lighten';
  for(const f of[1,.66,.33])for(let a=0;a<12;a++)c.drawImage(D,Math.cos(a*Math.PI/6)*r*f,Math.sin(a*Math.PI/6)*r*f);
  const o=mk(D.width,D.height);o.ctx.filter='blur('+Math.min(10,r*.4*Math.max(1,stretch)).toFixed(1)+'px)';o.ctx.drawImage(m.canvas,0,0);return o.canvas;
}
/* Depth Anything V2 (small) via transformers.js — runs on this machine, one inference at a time */
let depthPipe,depthQueue=Promise.resolve();
export async function estimateDepth(canvas){
  const T=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3');
  depthPipe??=(async()=>{
    const id='onnx-community/depth-anything-v2-small';
    if(navigator.gpu){try{return await T.pipeline('depth-estimation',id,{device:'webgpu',dtype:'fp16'})}catch(e){console.warn('[lab] webgpu unavailable, using wasm',e)}}
    return T.pipeline('depth-estimation',id,{device:'wasm',dtype:'q8'});
  })();
  const pipe=await depthPipe,run=depthQueue.then(()=>pipe(T.RawImage.fromCanvas(canvas)));depthQueue=run.catch(()=>{});
  let D=(await run).depth.toCanvas();
  if(!(D instanceof HTMLCanvasElement)){const m=mk(D.width,D.height);m.ctx.drawImage(D,0,0);D=m.canvas}
  return D;
}

/* ---------------------------------------------------------------- shaders */
const GLSL_COMMON=`
float lum(vec3 c){return dot(c,vec3(.299,.587,.114));}
float h21(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float a=.5,t=0.;for(int i=0;i<4;i++){t+=a*vnoise(p);p*=2.03;a*=.5;}return t;}
`;
const VS_SCENE=`#version 300 es
layout(location=0) in vec2 aUV;
uniform sampler2D uDepthA,uDepthB;
uniform mat4 uProj;uniform vec4 uHist[${HIST}];uniform vec2 uCoverB;
uniform float uCamZ,uAspect,uRelief,uMorph,uPointPx,uTime,uDissolve,uTurb,uWind,uSlit;uniform int uSlitAxis;
out vec2 vUV;out float vD;out float vK;
${GLSL_COMMON}
void main(){
  vec2 uvB=(aUV-.5)*uCoverB+.5;
  float d=mix(lum(textureLod(uDepthA,aUV,0.).rgb),lum(textureLod(uDepthB,uvB,0.).rgb),uMorph);
  vec3 p=vec3((aUV.x-.5)*uAspect,.5-aUV.y,(d-.5)*uRelief);
  float k=0.;
  if(uDissolve>0.){                                   // 05 particle dissolve: each point leaves at its own moment
    float r=mix(vnoise(aUV*vec2(7.,5.)+3.),h21(aUV*311.7),.22),t=uTime;   // mostly low-frequency: regions lift off together as wisps
    k=smoothstep(r*.55,r*.55+.45,uDissolve);
    vec3 n=vec3(sin(aUV.y*9.+t*.7+r*6.28),cos(aUV.x*8.+t*.6+r*5.),sin((aUV.x+aUV.y)*7.+t*.5+r*3.));
    p+=(n*uTurb*.6+vec3(uWind*.5,.1,.15))*k*k*(.4+r*.8);
  }
  float key=uSlitAxis==0?aUV.y:(uSlitAxis==1?aUV.x:1.-d);  // 07 slit-scan: rows / columns / depth layers lag behind the camera
  vec4 h=uHist[int(clamp(key*uSlit*${HIST-1}.,0.,${HIST-1}.))];
  float cy=cos(h.x),sy=sin(h.x),cp=cos(h.y),sp=sin(h.y);
  p=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);
  p=vec3(p.x,cp*p.y-sp*p.z,sp*p.y+cp*p.z);
  p.xy-=h.zw;p.z-=uCamZ;
  gl_Position=uProj*vec4(p,1.);
  gl_PointSize=max(1.,uPointPx/gl_Position.w*(1.-.6*k));
  vUV=aUV;vD=d;vK=k;
}`;
const FS_SCENE=`#version 300 es
precision highp float;
in vec2 vUV;in float vD;in float vK;
uniform sampler2D uColorA,uColorB,uVideo,uDepthA;
uniform vec2 uCoverB,uCoverV,uLightPos,uTexel;uniform vec3 uLightCol;
uniform float uMorph,uVideoMix,uLight,uLightH,uAmbient,uSpec,uAspect,uRelief,uPoints,uRound;
layout(location=0) out vec4 o0;layout(location=1) out vec4 o1;
${GLSL_COMMON}
void main(){
  if(uPoints>.5&&uRound>.5){vec2 q=gl_PointCoord-.5;if(dot(q,q)>.25)discard;}
  vec3 c=mix(texture(uColorA,vUV).rgb,texture(uColorB,(vUV-.5)*uCoverB+.5).rgb,uMorph);
  c=mix(c,texture(uVideo,(vUV-.5)*uCoverV+.5).rgb,uVideoMix);              // 12 video handoff
  if(uLight>0.){                                                           // 03 relight: normals from the depth map
    float dx=lum(texture(uDepthA,vUV+vec2(uTexel.x,0.)).rgb)-lum(texture(uDepthA,vUV-vec2(uTexel.x,0.)).rgb);
    float dy=lum(texture(uDepthA,vUV+vec2(0.,uTexel.y)).rgb)-lum(texture(uDepthA,vUV-vec2(0.,uTexel.y)).rgb);
    vec3 n=normalize(vec3(-dx*9.,dy*9.,1.));
    vec3 P=vec3((vUV.x-.5)*uAspect,.5-vUV.y,(vD-.5)*uRelief),L=vec3((uLightPos.x-.5)*uAspect,.5-uLightPos.y,uLightH);
    vec3 l=L-P;float dist=length(l);l/=dist;
    float att=1./(1.+dist*dist*2.2),diff=max(dot(n,l),0.)*att;
    float sp=pow(max(dot(reflect(-l,n),vec3(0.,0.,1.)),0.),24.)*att*uSpec;
    vec3 lit=c*(uAmbient+diff*2.6*uLightCol)+sp*uLightCol;
    c=mix(c,lit,uLight);
  }
  o0=vec4(c,1.);o1=vec4(vD,vUV,1.);
}`;
const VS_TEXT=`#version 300 es
uniform mat4 uProj;uniform vec4 uHist0;uniform vec4 uRect;uniform float uZ,uCamZ;   // uRect: cx, cy, w, h in plane units
out vec2 vUV;
void main(){
  vec2 q=vec2(float(gl_VertexID&1),float((gl_VertexID>>1)&1));
  vec3 p=vec3(uRect.x+(q.x-.5)*uRect.z,uRect.y+(q.y-.5)*uRect.w,uZ);
  float cy=cos(uHist0.x),sy=sin(uHist0.x),cp=cos(uHist0.y),sp=sin(uHist0.y);
  p=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);p=vec3(p.x,cp*p.y-sp*p.z,sp*p.y+cp*p.z);
  p.xy-=uHist0.zw;p.z-=uCamZ;
  gl_Position=uProj*vec4(p,1.);vUV=vec2(q.x,1.-q.y);
}`;
const FS_TEXT=`#version 300 es
precision highp float;
in vec2 vUV;uniform sampler2D uTex;uniform vec3 uCol;uniform float uD;
layout(location=0) out vec4 o0;layout(location=1) out vec4 o1;
void main(){if(texture(uTex,vUV).a<.5)discard;o0=vec4(uCol,1.);o1=vec4(uD,0.,0.,.5);}`;   // 06 type lives at a depth; the depth buffer does the occlusion
const VS_TRI=`#version 300 es
out vec2 vUV;
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2))*2.-1.;vUV=p*.5+.5;gl_Position=vec4(p,0.,1.);}`;
const FS_DOF=`#version 300 es
precision highp float;
in vec2 vUV;uniform sampler2D uCol,uAux;uniform float uOn,uFocus,uAperture;uniform vec2 uPx;
out vec4 o;
void main(){                                                                  // 01 rack focus: blur grows with distance from the focal plane
  vec3 c=texture(uCol,vUV).rgb;
  if(uOn<.5){o=vec4(c,1.);return;}
  float d=texture(uAux,vUV).r,R=min(abs(d-uFocus)*uAperture,1.)*26.;
  vec3 acc=c;float w=1.;
  for(int i=1;i<44;i++){
    float fi=float(i),r=sqrt(fi/44.)*26.,a=fi*2.39996;
    vec2 uv=vUV+vec2(cos(a),sin(a))*r*uPx;
    float sd=texture(uAux,uv).r,sR=min(abs(sd-uFocus)*uAperture,1.)*26.;
    float k=clamp((sd>d+.03?sR:min(sR,R))-r+1.5,0.,1.);                        // nearer samples spill by their own blur; farther ones never bleed onto a sharper pixel
    acc+=texture(uCol,uv).rgb*k;w+=k;
  }
  o=vec4(acc/w,1.);
}`;
const FS_COMP=`#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uCol,uAux,uPrev;
uniform vec2 uRes,uMouse,uMotion,uGrid;uniform float uTime,uDpr;
uniform float uSortOn,uSortThr,uSortLen;uniform vec2 uSortDir;
uniform float uFog,uFogNear,uFogFar,uFogSpeed,uFogTone;
uniform float uContour,uContourN,uContourSpeed,uContourDim;
uniform float uScanOn,uScan,uScanBand;
uniform float uLensOn,uLensR;uniform int uLensMode;
uniform float uMono,uDither,uDitherLv,uScanlines,uMosh,uMoshBlock;
out vec4 o;
${GLSL_COMMON}
const vec3 RED=vec3(.902,.098,.098),BGC=vec3(.039);
const float M[16]=float[16](0.,8.,2.,10.,12.,4.,14.,6.,3.,11.,1.,9.,15.,7.,13.,5.);
vec3 heat(float d){vec3 h=d<.5?mix(BGC,RED,d*2.):mix(RED,vec3(.92),(d-.5)*2.);return h*(.62+.38*step(.5,fract(d*28.)))+.06;}   // banded so flat ranges still show structure
void main(){
  vec2 px=1./uRes,uv=vUV;
  vec3 c=texture(uCol,uv).rgb;
  if(uSortOn>.5&&lum(c)>uSortThr){                                            // 08 pixel sort: inside every bright run, colours are re-laid dark to bright
    vec2 st=uSortDir*px*(1.+uSortLen*6.);float n=0.,m=0.;vec3 lo=c,hi=c;
    for(int i=1;i<44;i++){vec3 q=texture(uCol,uv-st*float(i)).rgb;if(lum(q)<uSortThr)break;n=float(i);if(lum(q)<lum(lo))lo=q;if(lum(q)>lum(hi))hi=q;}
    for(int i=1;i<44;i++){vec3 q=texture(uCol,uv+st*float(i)).rgb;if(lum(q)<uSortThr)break;m=float(i);if(lum(q)<lum(lo))lo=q;if(lum(q)>lum(hi))hi=q;}
    float run=n+m;c=mix(c,mix(lo,hi,n/(run+1.)),smoothstep(2.,10.,run));
  }
  vec4 ax=texture(uAux,uv);float d=ax.r,a=ax.a;
  if(a>0.){
    if(uFog>0.){                                                               // 04 fog sits between depth layers and drifts
      float far=1.-d,f=smoothstep(uFogNear,uFogFar,far)*(.45+.75*fbm(uv*vec2(3.,5.)+vec2(uTime*uFogSpeed*.15,far*3.)+far*2.));
      c=mix(c,vec3(uFogTone),clamp(f*uFog,0.,1.));
    }
    if(uContour>0.){                                                           // 10 contours: depth drawn as elevation lines
      float t=d*uContourN-uTime*uContourSpeed,w=fwidth(t)*1.2,l=1.-smoothstep(w,w*2.2,abs(fract(t)-.5));
      float major=1.-smoothstep(w,w*2.2,abs(fract(t/5.)-.5)*5.);
      c=mix(c*(1.-uContourDim*uContour),mix(vec3(.92),RED,major),max(l,major)*uContour);
    }
    if(uScanOn>.5){                                                            // 02 scan reveal: near to far
      float near=1.-d,vis=step(near,uScan),e=smoothstep(uScanBand,0.,abs(near-uScan));
      c=mix(BGC,c,vis)+RED*e*1.2;
    }
  }
  if(uLensOn>.5){                                                              // 09 lens: another reading of the same pixels
    vec2 fp=uv*uRes;float r=distance(fp,uMouse),R=uLensR*uRes.y;
    if(r<R){
      vec3 m=BGC;
      if(a>0.){
        if(uLensMode==0)m=heat(d);
        else if(uLensMode==1){vec2 g=abs(fract(ax.gb*uGrid)-.5);float wl=1.-smoothstep(.0,.08,min(g.x,g.y));m=mix(BGC,vec3(.92),wl)+heat(d)*.18;}
        else if(uLensMode==2){vec3 nn=normalize(vec3(-dFdx(d)*uRes.x*.06,-dFdy(d)*uRes.y*.06,1.));m=nn*.5+.5;}
        else if(uLensMode==3)m=vec3(d);
        else{float t=d*30.,w=fwidth(t)*1.2;m=mix(BGC,vec3(.92),1.-smoothstep(w,w*2.2,abs(fract(t)-.5)));}
      }
      c=m;
    }
    c=mix(c,RED,1.-smoothstep(0.,1.5*uDpr,abs(r-R)));
  }
  c=mix(c,vec3(lum(c))*.918,uMono);
  if(uDither>0.){ivec2 q=ivec2(mod(floor(gl_FragCoord.xy/(uDpr*1.5)),4.));float th=(M[q.x+q.y*4]+.5)/16.;c=mix(c,floor(c*uDitherLv+th)/uDitherLv,uDither);}
  c*=1.-uScanlines*.3*step(.5,fract(gl_FragCoord.y/(uDpr*4.)));
  if(uMosh>0.){                                                                // 08 datamosh: random blocks refuse to update and drag the last frame along
    vec2 b=floor(gl_FragCoord.xy/(uMoshBlock*uDpr));
    if(h21(b+floor(uTime*7.)*.13)<uMosh)c=texture(uPrev,clamp(uv-uMotion,0.,1.)).rgb;
  }
  o=vec4(c,1.);
}`;

/* ---------------------------------------------------------------- engine */
export function createLab(canvas,{camera=true}={}){
  const gl=canvas.getContext('webgl2',{antialias:false,alpha:false});
  if(!gl)throw new Error('WebGL2 unavailable');
  const floatOK=!!gl.getExtension('EXT_color_buffer_float');
  const S={...DEFAULTS};

  function program(vs,fs){
    const sh=(t,s)=>{const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(x)+'\n'+s.split('\n').map((l,i)=>(i+1)+': '+l).join('\n'));return x};
    const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
    const u={};for(let i=0,n=gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);i<n;i++){const nm=gl.getActiveUniform(p,i).name.replace(/\[0\]$/,'');u[nm]=gl.getUniformLocation(p,nm)}
    return{p,u};
  }
  const pScene=program(VS_SCENE,FS_SCENE),pText=program(VS_TEXT,FS_TEXT),pDof=program(VS_TRI,FS_DOF),pComp=program(VS_TRI,FS_COMP);
  const emptyVAO=gl.createVertexArray();

  function texFrom(src,mip){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,src);if(mip)gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,mip?gl.LINEAR_MIPMAP_LINEAR:gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t}
  const black=(()=>{const m=mk(2,2);m.ctx.fillStyle='#000';m.ctx.fillRect(0,0,2,2);return m.canvas})();

  /* render targets */
  let W=0,H=0,dpr=1,scene,tDof,ping=[],cur=0;
  function target(w,h,aux){
    const fb=gl.createFramebuffer(),mkTex=(ifmt,fmt,type)=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,ifmt,w,h,0,fmt,type,null);
      for(const k of[gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,k,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t};
    gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
    const col=mkTex(gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,col,0);
    let ax=null,rb=null;
    if(aux){ax=floatOK?mkTex(gl.RGBA16F,gl.RGBA,gl.HALF_FLOAT):mkTex(gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,gl.TEXTURE_2D,ax,0);
      rb=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,rb);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,w,h);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,rb);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1])}
    return{fb,col,ax,rb,free(){gl.deleteFramebuffer(fb);gl.deleteTexture(col);ax&&gl.deleteTexture(ax);rb&&gl.deleteRenderbuffer(rb)}};
  }
  function resize(){
    dpr=Math.min(devicePixelRatio||1,1.5);const w=Math.max(2,Math.round(canvas.clientWidth*dpr)),h=Math.max(2,Math.round(canvas.clientHeight*dpr));
    if(w===W&&h===H)return;W=w;H=h;canvas.width=w;canvas.height=h;
    [scene,tDof,...ping].forEach(t=>t&&t.free());scene=target(w,h,true);tDof=target(w,h);ping=[target(w,h),target(w,h)];
  }

  /* images: slot A is shown, slot B is the morph target */
  const img={A:null,B:null};let mesh=null,video=null,texVideo=texFrom(black),videoAspect=1,cpuDepth=null;
  function buildMesh(aspect){
    const rows=320,cols=Math.round(rows*aspect),uv=new Float32Array((rows+1)*(cols+1)*2),idx=new Uint32Array(rows*cols*6);
    for(let y=0,i=0;y<=rows;y++)for(let x=0;x<=cols;x++){uv[i++]=x/cols;uv[i++]=y/rows}
    for(let y=0,i=0;y<rows;y++)for(let x=0;x<cols;x++){const a=y*(cols+1)+x,b=a+1,c=a+cols+1,d=c+1;idx[i++]=a;idx[i++]=c;idx[i++]=b;idx[i++]=b;idx[i++]=c;idx[i++]=d}
    if(mesh){gl.deleteVertexArray(mesh.vao);gl.deleteBuffer(mesh.vb);gl.deleteBuffer(mesh.ib)}
    const vao=gl.createVertexArray(),vb=gl.createBuffer(),ib=gl.createBuffer();gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,uv,gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);gl.bindVertexArray(null);
    mesh={vao,vb,ib,rows,cols,verts:(rows+1)*(cols+1),indices:idx.length,aspect};
  }
  function setImage(slot,color,depth,{edge=1,level=.55}={}){
    const o=img[slot];if(o){gl.deleteTexture(o.tc);gl.deleteTexture(o.td)}
    const lv=levelDepth(depth,level),ready=soften(lv.canvas,edge,lv.stretch);
    img[slot]={tc:texFrom(color,true),td:texFrom(ready),aspect:color.width/color.height,color,depth,dw:depth.width,dh:depth.height,stretch:lv.stretch};
    if(slot==='A'){if(!mesh||Math.abs(mesh.aspect-img.A.aspect)>1e-4)buildMesh(img.A.aspect);
      const m=mk(256,Math.max(2,Math.round(256/img.A.aspect)));m.ctx.drawImage(ready,0,0,m.canvas.width,m.canvas.height);cpuDepth={w:m.canvas.width,h:m.canvas.height,d:m.ctx.getImageData(0,0,m.canvas.width,m.canvas.height).data}}
  }
  function setDepth(slot,depth,opt){if(img[slot])setImage(slot,img[slot].color,depth,opt)}
  function setVideo(el){video=el;if(el){videoAspect=(el.videoWidth||16)/(el.videoHeight||9)}}
  const cover=(target,src)=>src>target?[target/src,1]:[1,src/target];       // uv scale so `src` covers a rect of aspect `target`

  /* depth type */
  let textTex=null,textKey='',textAspect=4;
  function updateText(){
    const key=S.text;if(key===textKey)return;textKey=key;
    const fs=220,m=mk(8,8);m.ctx.font=`400 ${fs}px "Archivo Black","Helvetica Neue",Arial,sans-serif`;
    const w=Math.min(4096,Math.ceil(m.ctx.measureText(key||' ').width*.95+40)),c=mk(w,Math.round(fs*1.02));
    c.ctx.font=m.ctx.font;c.ctx.textBaseline='alphabetic';c.ctx.fillStyle='#fff';c.ctx.letterSpacing='-10px';c.ctx.fillText(key,16,fs*.86,w-32);
    textAspect=c.canvas.width/c.canvas.height;if(textTex)gl.deleteTexture(textTex);textTex=texFrom(c.canvas,true);
  }
  document.fonts?.load('400 100px "Archivo Black"').then(()=>{textKey='';});

  /* camera + input */
  const cam={yaw:0,pitch:0,zoom:1,cx:0,cy:0},hist=new Float32Array(HIST*4),mouse={x:.5,y:.5,over:false},gyro={x:0,y:0};
  let focusCur=.6,lastCam={yaw:0,pitch:0,cx:0,cy:0};
  const fitDist=(A,Af)=>Math.min(1,A/Af)/(2*TAN);
  function planeUV(mxN,myN){const A=img.A?.aspect||1.5,Af=W/H,cz=fitDist(A,Af)/cam.zoom,nx=mxN*2-1,ny=-(myN*2-1);return[clamp((cam.cx+nx*TAN*Af*cz)/A+.5,0,1),clamp(.5-(cam.cy+ny*TAN*cz),0,1)]}
  function depthAt(u,v){if(!cpuDepth)return .5;const x=Math.min(cpuDepth.w-1,Math.floor(u*cpuDepth.w)),y=Math.min(cpuDepth.h-1,Math.floor(v*cpuDepth.h));return cpuDepth.d[(y*cpuDepth.w+x)*4]/255}
  function clampPan(){const A=img.A?.aspect||1.5,k=Math.max(0,1-1/S.zoom);S.cx=clamp(S.cx,-A/2*k,A/2*k);S.cy=clamp(S.cy,-.5*k,.5*k)}
  function zoomAt(f,mxN=.5,myN=.5){const A=img.A?.aspect||1.5,Af=W/H,d0=fitDist(A,Af),nx=mxN*2-1,ny=-(myN*2-1),z1=clamp(S.zoom*f,.5,16),c0=d0/S.zoom,c1=d0/z1;
    const wx=S.cx+nx*TAN*Af*c0,wy=S.cy+ny*TAN*c0;S.zoom=z1;S.cx=wx-nx*TAN*Af*c1;S.cy=wy-ny*TAN*c1;clampPan();api.onInput?.()}
  const ptrs=new Map();let pinch=0;
  canvas.addEventListener('pointermove',e=>{const r=canvas.getBoundingClientRect();mouse.x=(e.clientX-r.left)/r.width;mouse.y=(e.clientY-r.top)/r.height;mouse.over=true;
    const pt=ptrs.get(e.pointerId);if(!pt||!api.camera)return;const dx=e.clientX-pt.x,dy=e.clientY-pt.y;pt.x=e.clientX;pt.y=e.clientY;
    if(ptrs.size===2){const[a,b]=[...ptrs.values()],d=Math.hypot(a.x-b.x,a.y-b.y);if(pinch)zoomAt(d/pinch,mouse.x,mouse.y);pinch=d}
    else if(e.shiftKey){const w=2*TAN*(fitDist(img.A?.aspect||1.5,W/H)/cam.zoom)/r.height;S.cx-=dx*w;S.cy+=dy*w;clampPan()}
    else{S.yaw=clamp(S.yaw+dx*.004,-.8,.8);S.pitch=clamp(S.pitch+dy*.004,-.55,.55)}api.onInput?.()});
  canvas.addEventListener('pointerleave',()=>mouse.over=false);
  canvas.addEventListener('pointerdown',e=>{if(!api.camera)return;canvas.setPointerCapture(e.pointerId);ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY});pinch=0});
  const up=e=>{ptrs.delete(e.pointerId);pinch=0};canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',up);
  canvas.addEventListener('wheel',e=>{if(!api.camera)return;e.preventDefault();zoomAt(Math.exp(-e.deltaY*(e.ctrlKey?.012:.0016)),mouse.x,mouse.y)},{passive:false});
  canvas.addEventListener('dblclick',()=>{if(!api.camera)return;S.yaw=S.pitch=S.cx=S.cy=0;S.zoom=1;api.onInput?.()});
  addEventListener('deviceorientation',e=>{if(e.gamma==null)return;gyro.x=clamp(e.gamma/45,-1,1);gyro.y=clamp((e.beta-45)/45,-1,1)});

  function persp(asp,n,f){const o=new Float32Array(16);o[0]=P11/asp;o[5]=P11;o[10]=(f+n)/(n-f);o[11]=-1;o[14]=2*f*n/(n-f);return o}
  const bindTex=(unit,t)=>{gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t)};

  /* frame */
  let last=performance.now(),t0=last,fps=60,raf=0;
  function frame(now){
    raf=requestAnimationFrame(frame);
    const dt=Math.min(.05,(now-last)/1000);last=now;fps=lerp(fps,1/Math.max(dt,1e-3),.05);
    resize();api.onFrame?.(now,dt);
    if(!img.A||!mesh){gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,W,H);gl.clearColor(...BG,1);gl.clear(gl.COLOR_BUFFER_BIT);return}
    const time=(now-t0)/1000,k=1-Math.exp(-dt*9),A=img.A.aspect,Af=W/H,B=img.B||img.A;

    /* camera: state + sway + gyro, damped */
    const sw=S.sway,ss=S.swaySpeed,gy=S.gyroOn?1:0;
    const ty=S.yaw+Math.sin(time*ss)*sw*.3+gyro.x*.35*gy,tp=S.pitch+Math.cos(time*ss*.8)*sw*.14+gyro.y*.25*gy;
    cam.yaw=lerp(cam.yaw,ty,k);cam.pitch=lerp(cam.pitch,tp,k);cam.zoom=lerp(cam.zoom,S.zoom,k);cam.cx=lerp(cam.cx,S.cx,k);cam.cy=lerp(cam.cy,S.cy,k);
    hist.copyWithin(4,0,(HIST-1)*4);hist[0]=cam.yaw;hist[1]=cam.pitch;hist[2]=cam.cx;hist[3]=cam.cy;
    const camZ=fitDist(A,Af)/cam.zoom,points=S.geometry==='points';
    const motion=[(cam.yaw-lastCam.yaw)*.9+(cam.cx-lastCam.cx)/A,-(cam.pitch-lastCam.pitch)*.9+(cam.cy-lastCam.cy)];lastCam={...cam};

    const mUV=mouse.over?planeUV(mouse.x,mouse.y):[S.lightX,S.lightY];
    if(S.focusOn)focusCur=lerp(focusCur,S.focusFollow&&mouse.over?depthAt(...planeUV(mouse.x,mouse.y)):S.focus,1-Math.exp(-dt*6));
    const lightUV=S.lightFollow&&mouse.over?mUV:[S.lightX,S.lightY];
    const vOK=S.videoOn&&video&&video.readyState>=2;
    if(vOK){bindTex(4,texVideo);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,video);videoAspect=video.videoWidth/video.videoHeight}
    const vMix=vOK?S.videoMix:0,dis=S.dissolveOn?S.dissolve:0;

    /* pass 1: scene */
    gl.bindFramebuffer(gl.FRAMEBUFFER,scene.fb);gl.viewport(0,0,W,H);gl.enable(gl.DEPTH_TEST);
    gl.clearBufferfv(gl.COLOR,0,[...BG,1]);gl.clearBufferfv(gl.COLOR,1,[0,0,0,0]);gl.clearBufferfv(gl.DEPTH,0,[1]);
    let u=pScene.u;gl.useProgram(pScene.p);
    gl.uniformMatrix4fv(u.uProj,false,persp(Af,.02,20));gl.uniform4fv(u.uHist,hist);
    gl.uniform2fv(u.uCoverB,cover(A,B.aspect));gl.uniform2fv(u.uCoverV,cover(A,videoAspect));
    gl.uniform1f(u.uCamZ,camZ);gl.uniform1f(u.uAspect,A);gl.uniform1f(u.uRelief,S.relief*(1-vMix));gl.uniform1f(u.uMorph,S.morph);
    gl.uniform1f(u.uPointPx,(1/mesh.rows)*S.fill*P11*H*.5);gl.uniform1f(u.uTime,time);
    gl.uniform1f(u.uDissolve,dis);gl.uniform1f(u.uTurb,S.turb);gl.uniform1f(u.uWind,S.wind);
    gl.uniform1f(u.uSlit,S.slitOn?S.slit:0);gl.uniform1i(u.uSlitAxis,S.slitAxis|0);
    gl.uniform1f(u.uVideoMix,vMix);gl.uniform1f(u.uLight,S.lightOn?S.light:0);gl.uniform1f(u.uLightH,S.lightH);gl.uniform1f(u.uAmbient,S.ambient);gl.uniform1f(u.uSpec,S.spec);
    gl.uniform2fv(u.uLightPos,lightUV);gl.uniform2f(u.uTexel,2.5/img.A.dw,2.5/img.A.dh);
    gl.uniform3f(u.uLightCol,1,lerp(1,.78,S.lightWarm),lerp(1,.52,S.lightWarm));
    gl.uniform1f(u.uPoints,points?1:0);gl.uniform1f(u.uRound,S.round);
    gl.uniform1i(u.uColorA,0);gl.uniform1i(u.uColorB,1);gl.uniform1i(u.uDepthA,2);gl.uniform1i(u.uDepthB,3);gl.uniform1i(u.uVideo,4);
    bindTex(0,img.A.tc);bindTex(1,B.tc);bindTex(2,img.A.td);bindTex(3,B.td);bindTex(4,texVideo);
    gl.bindVertexArray(mesh.vao);
    if(points)gl.drawArrays(gl.POINTS,0,mesh.verts);else gl.drawElements(gl.TRIANGLES,mesh.indices,gl.UNSIGNED_INT,0);

    if(S.textOn&&S.text){                                                     // 06 depth type
      updateText();u=pText.u;gl.useProgram(pText.p);
      const h=S.textSize,w=Math.min(h*textAspect,A*.97);                  // never wider than the plate
      gl.uniformMatrix4fv(u.uProj,false,persp(Af,.02,20));gl.uniform4f(u.uHist0,hist[0],hist[1],hist[2],hist[3]);
      gl.uniform4f(u.uRect,0,.5-S.textY,w,w/textAspect);gl.uniform1f(u.uZ,(S.textDepth-.5)*S.relief);gl.uniform1f(u.uCamZ,camZ);gl.uniform1f(u.uD,S.textDepth);
      gl.uniform3fv(u.uCol,[[.918,.918,.918],[.902,.098,.098],[.039,.039,.039]][S.textColor|0]);
      gl.uniform1i(u.uTex,0);bindTex(0,textTex);gl.bindVertexArray(emptyVAO);gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    }

    /* pass 2: focus */
    gl.disable(gl.DEPTH_TEST);gl.bindVertexArray(emptyVAO);
    gl.bindFramebuffer(gl.FRAMEBUFFER,tDof.fb);u=pDof.u;gl.useProgram(pDof.p);
    gl.uniform1i(u.uCol,0);gl.uniform1i(u.uAux,1);bindTex(0,scene.col);bindTex(1,scene.ax);
    gl.uniform1f(u.uOn,S.focusOn?1:0);gl.uniform1f(u.uFocus,focusCur);gl.uniform1f(u.uAperture,S.aperture);gl.uniform2f(u.uPx,dpr/W,dpr/H);
    gl.drawArrays(gl.TRIANGLES,0,3);

    /* pass 3: composite (ping-pong so datamosh can read the last frame) */
    const out=ping[cur],prev=ping[1-cur];cur=1-cur;
    gl.bindFramebuffer(gl.FRAMEBUFFER,out.fb);u=pComp.u;gl.useProgram(pComp.p);
    gl.uniform1i(u.uCol,0);gl.uniform1i(u.uAux,1);gl.uniform1i(u.uPrev,2);bindTex(0,tDof.col);bindTex(1,scene.ax);bindTex(2,prev.col);
    const lensUV=S.lensFollow&&mouse.over?[mouse.x,mouse.y]:[S.lensX,S.lensY];
    gl.uniform2f(u.uRes,W,H);gl.uniform2f(u.uMouse,lensUV[0]*W,(1-lensUV[1])*H);gl.uniform2fv(u.uMotion,motion);gl.uniform2f(u.uGrid,mesh.cols/8,mesh.rows/8);
    gl.uniform1f(u.uTime,time);gl.uniform1f(u.uDpr,dpr);
    const dir=[[0,1],[1,0],[0,-1],[-1,0]][S.sortDir|0];
    gl.uniform1f(u.uSortOn,S.sortOn?1:0);gl.uniform1f(u.uSortThr,S.sortThr);gl.uniform1f(u.uSortLen,S.sortLen);gl.uniform2fv(u.uSortDir,dir);
    gl.uniform1f(u.uFog,S.fogOn?S.fog:0);gl.uniform1f(u.uFogNear,S.fogNear);gl.uniform1f(u.uFogFar,S.fogFar);gl.uniform1f(u.uFogSpeed,S.fogSpeed);gl.uniform1f(u.uFogTone,S.fogTone);
    gl.uniform1f(u.uContour,S.contourOn?S.contour:0);gl.uniform1f(u.uContourN,S.contourN);gl.uniform1f(u.uContourSpeed,S.contourSpeed*10);gl.uniform1f(u.uContourDim,S.contourDim);
    gl.uniform1f(u.uScanOn,S.scanOn?1:0);gl.uniform1f(u.uScan,S.scanAuto>0?(time*S.scanAuto)%1.35-.1:S.scan*1.1);gl.uniform1f(u.uScanBand,S.scanBand);
    gl.uniform1f(u.uLensOn,S.lensOn?1:0);gl.uniform1f(u.uLensR,S.lensR);gl.uniform1i(u.uLensMode,S.lensMode|0);
    gl.uniform1f(u.uMono,S.mono);gl.uniform1f(u.uDither,S.ditherOn?S.dither:0);gl.uniform1f(u.uDitherLv,S.ditherLv);gl.uniform1f(u.uScanlines,S.scanlines);
    gl.uniform1f(u.uMosh,S.moshOn?S.mosh:0);gl.uniform1f(u.uMoshBlock,S.moshBlock);
    gl.drawArrays(gl.TRIANGLES,0,3);

    gl.bindFramebuffer(gl.READ_FRAMEBUFFER,out.fb);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);
    gl.blitFramebuffer(0,0,W,H,0,0,W,H,gl.COLOR_BUFFER_BIT,gl.NEAREST);
  }

  const api={S,camera,setImage,setDepth,setVideo,zoomAt,depthAt,planeUV,
    get image(){return img},get mouse(){return mouse},
    info:()=>({fps:Math.round(fps),w:W,h:H,tris:mesh?mesh.indices/3:0,pts:mesh?mesh.verts:0,float:floatOK,focus:focusCur,hasVideo:!!video}),
    reset(){Object.assign(S,DEFAULTS)},start(){if(!raf)raf=requestAnimationFrame(frame)},
    async enableGyro(){if(typeof DeviceOrientationEvent!=='undefined'&&DeviceOrientationEvent.requestPermission){return (await DeviceOrientationEvent.requestPermission())==='granted'}return true}};
  return api;
}
