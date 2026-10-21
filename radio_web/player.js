'use strict';
const audio=document.getElementById('audio'), status=document.getElementById('status');
let enabled=false,current='',loading=false,program=null;
function select(p,now){
 if(!p||!Array.isArray(p.clips)||p.clips.length>32||!Number.isFinite(p.start_at))return null;
 let offset=now-p.start_at;if(offset<0)return null;
 for(const [index,clip] of p.clips.entries()){
  if(!/^[a-f0-9]{64}$/.test(clip.id)||!Number.isFinite(clip.duration)||clip.duration<=0)return null;
  if(offset<clip.duration)return {clip,offset,index};offset-=clip.duration;
 }return null;
}
function sync(){
 const chosen=select(program,Date.now()/1000);
 if(!chosen){audio.hidden=true;status.textContent='Off air';audio.pause();audio.removeAttribute('src');current='';return;}
 audio.hidden=false;
 status.textContent=(chosen.clip.kind==='speech'?'Ina speaking — ':'Now playing — ')+chosen.clip.title;
 if(!enabled)return;
 const key=program.revision+':'+chosen.index;
 // Clip occurrences have distinct timeline starts, even when the same song repeats.
 const source='/media/'+chosen.clip.id+'.opus';
 if(audio.getAttribute('src')!==source||current!==key){
  current=key;audio.preload='metadata';
  audio.onloadedmetadata=()=>{audio.currentTime=chosen.offset;audio.play().catch(()=>{status.textContent='Press play to listen.';});};
  audio.src=source;audio.load();
 }else if(!audio.paused&&Math.abs(audio.currentTime-chosen.offset)>4){audio.currentTime=chosen.offset;}
}
async function refresh(){if(loading)return;loading=true;try{
 const response=await fetch('/station.json',{cache:'no-store',credentials:'omit'});
 if(!response.ok)throw Error();const text=await response.text();if(text.length>32000)throw Error();
 program=JSON.parse(text);sync();
 }catch{program=null;sync();}finally{loading=false;}}
document.getElementById('listen').onclick=()=>{enabled=true;current='';refresh();};
audio.onended=()=>{current='';sync();};
setInterval(sync,1000);setInterval(refresh,15000);refresh();
