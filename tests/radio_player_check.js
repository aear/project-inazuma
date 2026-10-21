const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
let loads=0,requestedLoads=0,attributes={};
const audio={paused:true,currentTime:0,load(){requestedLoads++},pause(){this.paused=true},play(){this.paused=false;return Promise.resolve()},
 getAttribute(k){return attributes[k]||null},removeAttribute(k){delete attributes[k]}};
Object.defineProperty(audio,'src',{set(v){loads++;attributes.src=v},get(){return attributes.src}});
const status={textContent:''};Object.defineProperty(status,'innerHTML',{set(){throw Error('unsafe HTML')}});
const context={document:{getElementById(id){return id==='audio'?audio:id==='status'?status:{}}},
 Date:{now:()=>1000000},setInterval(){},fetch:async()=>({ok:true,text:async()=>'{"clips":[]}'})};
vm.createContext(context);
let code=fs.readFileSync(process.argv[2],'utf8').replace('setInterval(sync,1000);setInterval(refresh,15000);refresh();','');
vm.runInContext(code,context);
const id='a'.repeat(64),clip={id,title:'<script>must stay text</script>',kind:'speech',duration:10};
const p={revision:'r',start_at:995,clips:[clip,clip]};
context.p=p;
assert.equal(vm.runInContext('select(p,1000).index',context),0);
assert.equal(vm.runInContext('select(p,1006).index',context),1);
assert.equal(vm.runInContext('select(p,1020)',context),null);
assert.equal(vm.runInContext('select(p,900)',context),null);
vm.runInContext('program=p;enabled=true;sync();',context);assert.equal(loads,1);
vm.runInContext('sync();sync();',context);assert.equal(loads,1,'same occurrence must not reload each tick');
assert.equal(audio.preload,'metadata');assert.equal(requestedLoads,1,'metadata must be loaded before waiting for it');
assert.ok(status.textContent.includes('<script>must stay text</script>'));
context.p={...p,clips:[{...clip,id:'../../private'}]};
assert.equal(vm.runInContext('select(p,1000)',context),null);
console.log('Player timeline, repeat occurrence, hostile ID/text and stable playback checks passed.');
