/* DEPTH//LAB — loading helpers shared by both pages */
import {loadImg,toCanvas,lumaDepth} from './engine.js';

const cache=new Map();
/* library item -> {color, depth, src} canvases (depth falls back to luminance when the file is missing) */
export async function loadItem(item){
  if(cache.has(item.id))return cache.get(item.id);
  const color=toCanvas(await loadImg(item.src),2048);let depth,src='FILE';
  try{depth=toCanvas(await loadImg(item.depth),1280)}catch{depth=lumaDepth(color);src='LUMINANCE'}
  const r={color,depth,src,edge:item.edge||1};cache.set(item.id,r);return r;
}
export const fileToCanvas=async(file,max=2048)=>{const url=URL.createObjectURL(file);try{return toCanvas(await loadImg(url),max)}finally{URL.revokeObjectURL(url)}};
export function fileToVideo(file){return new Promise((ok,no)=>{const v=document.createElement('video');v.muted=true;v.loop=true;v.playsInline=true;v.src=URL.createObjectURL(file);
  v.onloadeddata=()=>{v.play();ok(v)};v.onerror=()=>no(new Error('video could not be read'))})}
export const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
